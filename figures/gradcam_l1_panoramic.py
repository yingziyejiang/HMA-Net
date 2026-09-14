
import argparse
import importlib.util
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAY_MEAN = np.array([0.485, 0.456, 0.406])
GRAY_STD = np.array([0.229, 0.224, 0.225])

def load_module(path):
    spec = importlib.util.spec_from_file_location('hma_model', os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def build_model(mod, ckpt, dev):
    model = mod.HMAAgePredictorV10(l1_backbone='resnet50', l3_backbone='resnet34', use_coral=True).to(dev)
    state = torch.load(ckpt, map_location=dev)
    model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)
    model.eval()
    return model

def find_l1_layer(model):
    feats = model.l1_encoder.features
    tgt = None

    def _scan(module):
        nonlocal tgt
        for m in module.children():
            if isinstance(m, nn.Conv2d):
                tgt = m
            elif len(list(m.children())) > 0:
                _scan(m)

    _scan(feats)
    if tgt is None and len(list(feats.children())) > 0:
        last = list(feats.children())[-1]
        if isinstance(last, nn.Conv2d):
            tgt = last
    return tgt

class GradCAML1:
    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, batch):
        self.model.zero_grad()
        out = self.model(batch)
        age = out[0]
        age.sum().backward()
        pg = self.grads.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((pg * self.acts).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, age, out[1]

def denorm(t):
    im = t.squeeze(0).cpu().permute(1, 2, 0).numpy() * GRAY_STD + GRAY_MEAN
    return np.clip(im, 0, 1)

def main():
    ap = argparse.ArgumentParser(description='L1 panoramic Grad-CAM for HMA-Net.')
    ap.add_argument('--train-py', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       '..', 'hma_net', 'hma_net.py'))
    ap.add_argument('--csv', required=True, help='manifest with img_path,label_path,age,gender,split')
    ap.add_argument('--labels-root', default='.', help='prefix prepended to the paths in the CSV')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--outdir', default='gradcam_l1')
    ap.add_argument('--n', type=int, default=100, help='number of test samples to render')
    ap.add_argument('--split', default='test')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    os.makedirs(args.outdir, exist_ok=True)
    mod = load_module(args.train_py)
    df = pd.read_csv(args.csv)
    test = df[df['split'] == args.split].reset_index(drop=True)
    print('samples:', len(test))
    ds = mod.DentalAgeDatasetV10(test, training=False)
    model = build_model(mod, args.ckpt, dev)
    layer = find_l1_layer(model)
    print('L1 target layer:', layer)
    gc = GradCAML1(model, layer)
    meta = []
    for i in range(min(args.n, len(ds))):
        batch = ds[i]
        b_in = {k: v.unsqueeze(0).to(dev) for k, v in batch.items()
                if k in ('img', 'teeth', 'pos_bbox', 'fdi', 'mask', 'gender')}
        row = test.iloc[i]
        with torch.enable_grad():
            cam, age_pred, _ = gc(b_in)
        age_v = float(age_pred.item() if hasattr(age_pred, 'item') else age_pred)
        cam_np = F.interpolate(cam, size=(256, 256), mode='bilinear',
                               align_corners=False).squeeze().cpu().numpy()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
        ax.imshow(denorm(b_in['img']))
        ax.imshow(cam_np, cmap='jet', alpha=0.5)
        ax.axis('off')
        ax.set_title('idx=%d age_t=%.1f age_p=%.1f g=%d' % (i, row['age'], age_v, int(row['gender'])), fontsize=8)
        png = os.path.join(args.outdir, 'l1_idx%03d.png' % i)
        fig.savefig(png, bbox_inches='tight')
        plt.close(fig)
        meta.append({'idx': i, 'img_path': row['img_path'], 'age_true': float(row['age']),
                     'age_pred': age_v, 'gender': int(row['gender']), 'png': png})
        np.savez(os.path.join(args.outdir, 'l3_idx%03d.npz' % i),
                 **{k: batch[k].numpy() for k in ('teeth', 'pos_bbox', 'fdi', 'mask')})
        if i % 10 == 0:
            print('  %d/%d done' % (i, args.n), flush=True)
    import json
    json.dump(meta, open(os.path.join(args.outdir, 'meta.json'), 'w'), indent=2)
    print('DONE', len(meta))

if __name__ == '__main__':
    main()
