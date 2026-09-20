
import argparse
import importlib.util
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_module(path):
    spec = importlib.util.spec_from_file_location('hma_model', os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def build_model(mod, ckpt, dev):
    model = mod.HMAAgePredictorV10(l1_backbone='resnet50', l3_backbone='resnet34',
                                   use_coral=True).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state)
    model.eval()
    return model

def read_obb(path):
    obb = {}
    for line in open(path):
        p = line.split()
        if len(p) == 6:
            obb[int(p[0])] = tuple(float(v) for v in p[1:])
    return obb

def main():
    ap = argparse.ArgumentParser(description='Per-tooth Grad-CAM with global normalisation.')
    ap.add_argument('--train-py', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       '..', 'hma_net', 'hma_net.py'))
    ap.add_argument('--csv', required=True)
    ap.add_argument('--labels-root', default='.', help='prefix prepended to the paths in the CSV')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--index', type=int, default=0, help='row index within the selected split')
    ap.add_argument('--split', default='test')
    ap.add_argument('--out', default='gradcam_l3_per_tooth.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--crop-factor', type=float, default=1.4 * 1.6,
                    help='how much larger than the box the 64x64 tile of the CAM is (default 1.4*1.6)')
    ap.add_argument('--vmax-pct', type=float, default=98.0, help='upper percentile of the global scale')
    ap.add_argument('--pmin-pct', type=float, default=10.0, help='lower percentile of the global scale')
    ap.add_argument('--alpha-base', type=float, default=0.30)
    ap.add_argument('--alpha-scale', type=float, default=0.15, help='extra alpha for the strongest tooth')
    ap.add_argument('--line-width', type=int, default=2)
    args = ap.parse_args()

    dev = torch.device(args.device)
    mod = load_module(args.train_py)
    df = pd.read_csv(args.csv)
    te = df[df['split'] == args.split].reset_index(drop=True)
    row = te.iloc[args.index]
    img = cv2.imread(os.path.join(args.labels_root, row['img_path']))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]
    print('full W,H:', W, H)
    obb = read_obb(os.path.join(args.labels_root, row['label_path']))

    model = build_model(mod, args.ckpt, dev)
    ds = mod.DentalAgeDatasetV10(te, img_size=256, jaw_mode='crop', training=False)
    sample = ds[args.index]
    mask0 = sample['mask'].clone()
    fdi = sample['fdi'].numpy()
    valid = [t for t in range(sample['teeth'].shape[0]) if mask0[t] > 0 and int(fdi[t]) in obb]
    print('valid teeth:', len(valid))

    last_conv = None
    for mm in model.l3_encoder.bb.modules():
        if isinstance(mm, nn.Conv2d):
            last_conv = mm
    acts = {}
    last_conv.register_forward_hook(lambda m, i, o: acts.__setitem__('x', o))

    def run_forward(mask_t):
        batch = {}
        for k, v in sample.items():
            batch[k] = (torch.as_tensor(mask_t).unsqueeze(0) if k == 'mask'
                        else torch.as_tensor(v).unsqueeze(0)).to(dev)
        model.zero_grad()
        with torch.enable_grad():
            out = model(batch)
            age = out[0]
        act = acts['x'].detach().clone()
        grad = torch.autograd.grad(age, acts['x'], retain_graph=True)[0].detach().clone()
        return age.item(), grad, act

    tooth_cams, tooth_energy = {}, {}
    for t in valid:
        m_t = torch.zeros_like(mask0)
        m_t[t] = 1.0
        _, grad_t, act_t = run_forward(m_t)
        cam = F.relu((grad_t[t].unsqueeze(0) * act_t[t].unsqueeze(0)).sum(dim=1, keepdim=True))[0, 0].numpy()
        cam = cv2.resize(cam, (64, 64), interpolation=cv2.INTER_LINEAR)
        tooth_cams[t] = cam
        tooth_energy[t] = float(cam.sum())

    allcams = np.stack([tooth_cams[t] for t in valid])
    vmax = np.percentile(allcams, args.vmax_pct)
    pmin = np.percentile(allcams, args.pmin_pct)
    print('global cam min=%.4f max=%.4f  vmax=%.4f pmin=%.4f' % (allcams.min(), allcams.max(), vmax, pmin))
    energies = np.array([tooth_energy[t] for t in valid])
    emax, emin = energies.max(), energies.min()

    fraction = 1.0 / args.crop_factor
    overlay = img_rgb.astype(np.float32) / 255.0

    def box_geometry(fid):
        cx, cy, cw, ch, ang = obb[fid]
        cx_i, cy_i, w_i, h_i = cx * W, cy * H, cw * W, ch * H
        ra = np.deg2rad(ang)
        c, s = np.cos(ra), np.sin(ra)
        hw, hh = w_i / 2, h_i / 2
        return cx_i, cy_i, hw, hh, c, s

    for t in valid:
        fid = int(fdi[t])
        cam_g = np.clip((tooth_cams[t] - pmin) / (vmax - pmin + 1e-12), 0, 1)
        cx_i, cy_i, hw, hh, c, s = box_geometry(fid)
        n = cam_g.shape[0]
        half = int(n * fraction / 2)
        cc = n // 2
        cam_tooth = cam_g[cc - half:cc + half, cc - half:cc + half]
        pts = [(cx_i + sx * hw * c - sy * hh * s, cy_i + sx * hw * s + sy * hh * c)
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        xmin = min(p[0] for p in pts); xmax = max(p[0] for p in pts)
        ymin = min(p[1] for p in pts); ymax = max(p[1] for p in pts)
        bw, bh = int(xmax - xmin), int(ymax - ymin)
        if bw <= 2 or bh <= 2:
            continue
        x0, y0 = int(xmin), int(ymin)
        x0c, y0c = max(0, x0), max(0, y0)
        x1c, y1c = min(W, x0 + bw), min(H, y0 + bh)
        cam_r = cv2.resize(cam_tooth.astype(np.float32), (bw, bh))
        cam_col = plt.cm.jet(np.clip(cam_r, 0, 1))[:, :, :3]
        yy, xx = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float32)
        dx, dy = xx - cx_i, yy - cy_i
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        in_box = (np.abs(lx) <= hw) & (np.abs(ly) <= hh)
        region = overlay[y0c:y1c, x0c:x1c].copy()
        cam_slice = cam_col[y0c - y0:y1c - y0, x0c - x0:x1c - x0]
        gnorm = (tooth_energy[t] - emin) / (emax - emin + 1e-12)
        alpha = args.alpha_base + args.alpha_scale * gnorm
        overlay[y0c:y1c, x0c:x1c] = np.where(in_box[..., None],
                                             (1 - alpha) * region + alpha * cam_slice, region)

    for t in valid:
        cx_i, cy_i, hw, hh, c, s = box_geometry(int(fdi[t]))
        pts = [(int(cx_i + sx * hw * c - sy * hh * s), int(cy_i + sx * hw * s + sy * hh * c))
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        cv2.polylines(overlay, np.array([pts], np.int32), True, (1, 1, 1), args.line_width)

    out = np.uint8(np.clip(overlay, 0, 1) * 255)
    cv2.imwrite(args.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print('saved', args.out, out.shape)

if __name__ == '__main__':
    main()
