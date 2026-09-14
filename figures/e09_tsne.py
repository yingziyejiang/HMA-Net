import argparse, os, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hma_net import DentalAgeDataset, HMAAgePredictor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

FEATS = {}


def make_hook(name, mode='direct'):
    def fn(module, inp, out):
        if mode == 'masked':
            FEATS[name] = out.detach().float()
        else:
            FEATS[name] = out.detach().float().flatten(1)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--outdir', default='figs_out')
    ap.add_argument('--split', default='test')
    ap.add_argument('--perplexity', type=float, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(a.csv)
    part = df[df.split == a.split]
    print('test rows:', len(part), flush=True)
    ds = DentalAgeDataset(part, training=False)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = HMAAgePredictor().to(a.device).eval()
    ck = torch.load(a.ckpt, map_location=a.device)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    names = [n for n, _ in model.named_modules()]
    target = {}
    for want, key in (('l1_encoder', 'l1_encoder'), ('l3_aggr', 'l3_aggr'), ('age_fusion_mlp', 'age_fusion_mlp')):
        if want in names:
            target[want] = True
        else:
            print('WARN missing module:', want, '| candidates:', [x for x in names if x.startswith(want[:6])][:4])
    model.l1_encoder.register_forward_hook(make_hook('l1'))
    model.l3_aggr.register_forward_hook(make_hook('l3', mode='masked'))
    if 'age_fusion_mlp' in target:
        model.age_fusion_mlp.register_forward_hook(make_hook('fused'))

    L1, L3, FU, ages, sexes = [], [], [], [], []
    with torch.no_grad():
        for batch in ld:
            b = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if b['img'].shape[1] != 3 and b['img'].shape[-1] == 3:
                b['img'] = b['img'].permute(0, 3, 1, 2)
            b['img'] = b['img'].contiguous()
            if b['teeth'].shape[2] != 3 and b['teeth'].shape[-1] == 3:
                b['teeth'] = b['teeth'].permute(0, 1, 4, 2, 3).contiguous()
            FEATS.clear()
            model(b)
            mask = b['mask'].unsqueeze(-1)
            l3 = (FEATS['l3'] * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            L1.append(FEATS['l1'].cpu().numpy())
            L3.append(l3.cpu().numpy())
            if 'fused' in FEATS:
                FU.append(FEATS['fused'].cpu().numpy())
            ages.append(b['age'].cpu().numpy().ravel())
            sexes.append(b['gender'].cpu().numpy().ravel())
    L1 = np.concatenate(L1); L3 = np.concatenate(L3)
    FU = np.concatenate(FU) if FU else None
    ages = np.concatenate(ages); sexes = np.concatenate(sexes)
    print('features:', L1.shape, L3.shape, None if FU is None else FU.shape, flush=True)

    sets = [('L1 (panoramic)', L1), ('L3 (tooth)', L3)]
    if FU is not None:
        sets.append(('L1+L3 (fused)', FU))
    embs, sels = [], []
    for name, X in sets:
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
        sel = np.random.RandomState(a.seed).choice(len(Xs), min(len(Xs), 2000), replace=False)
        sels.append(sel)
        e = TSNE(n_components=2, perplexity=a.perplexity, random_state=a.seed,
                 init='pca', learning_rate='auto').fit_transform(Xs[sel])
        embs.append(e)
        print('tsne done:', name, e.shape, flush=True)

    plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), constrained_layout=True)
    letters = 'abcdef'
    for col, (name, (e, sel)) in enumerate(zip(sets, zip(embs, sels))):
        ax = axes[0][col]
        s0 = ax.scatter(e[:, 0], e[:, 1], c=ages[sel], cmap='viridis', s=6, linewidths=0)
        ax.set_title(name, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('t-SNE 1', fontsize=8)
        if col == 0:
            ax.set_ylabel('t-SNE 2', fontsize=8)
        ax.text(-0.03, 1.06, '(%s)' % letters[col], transform=ax.transAxes, fontsize=11, va='bottom')
        if col == 2:
            cb = fig.colorbar(s0, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label('chronological age (yr)', fontsize=8)
        ax = axes[1][col]
        m = sexes[sel] > 0.5
        ax.scatter(e[~m, 0], e[~m, 1], c='#2c7fb8', s=6, linewidths=0, label='female')
        ax.scatter(e[m, 0], e[m, 1], c='#e6550d', s=6, linewidths=0, label='male')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel('t-SNE 1', fontsize=8)
        if col == 0:
            ax.set_ylabel('t-SNE 2', fontsize=8)
        ax.text(-0.03, 1.06, '(%s)' % letters[3 + col], transform=ax.transAxes, fontsize=11, va='bottom')
        if col == 2:
            ax.legend(loc='best', fontsize=8, frameon=False, markerscale=2)
    for row, lab in ((0, 'coloured by chronological age'), (1, 'coloured by sex')):
        axes[row][0].annotate(lab, xy=(0, 0.5), xytext=(-0.13, 0.5), xycoords='axes fraction',
                              textcoords='axes fraction', rotation=90, va='center', ha='right', fontsize=9)
    png = os.path.join(a.outdir, 'e09_tsne_age_sex.png')
    pdf = os.path.join(a.outdir, 'e09_tsne_age_sex.pdf')
    fig.savefig(png, dpi=300); fig.savefig(pdf)
    np.savez(os.path.join(a.outdir, 'e09_embeddings.npz'),
             **{'l1': L1, 'l3': L3, 'fused': FU if FU is not None else np.zeros(1),
                'age': ages, 'sex': sexes})
    print('saved:', png, pdf, flush=True)


if __name__ == '__main__':
    main()
