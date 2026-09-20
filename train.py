"""Minimal training entry point for HMA-Net (two-phase, early stopping)."""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hma_net'))
from hma_net import (BalancedBatchSampler, DentalAgeDataset, HMAAgePredictor,
                     compute_loss, evaluate, set_trainable)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True, help='CSV with img_path,label_path,age,gender,split')
    p.add_argument('--outdir', default='runs/hma_net')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--phase1-epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr1', type=float, default=1e-3)
    p.add_argument('--lr2', type=float, default=1e-4)
    p.add_argument('--lr-halve-every', type=int, default=20)
    p.add_argument('--gender-weight', type=float, default=3.0)
    p.add_argument('--coral-weight', type=float, default=0.5)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--min-epochs', type=int, default=30)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def loaders(manifest, batch_size, num_workers):
    import pandas as pd
    df = pd.read_csv(manifest)
    tr = DentalAgeDataset(df[df.split == 'train'], training=True)
    va = DentalAgeDataset(df[df.split == 'val'], training=False)
    te = DentalAgeDataset(df[df.split == 'test'], training=False)
    return (DataLoader(tr, batch_sampler=BalancedBatchSampler(tr, batch_size), num_workers=num_workers),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers))


def main():
    a = parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    train_ld, val_ld, test_ld = loaders(a.manifest, a.batch_size, a.num_workers)
    model = HMAAgePredictor().to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=a.lr_halve_every, gamma=0.5)
    best = {'age': float('inf'), 'sex': -1.0}
    best_epoch = {'age': 0, 'sex': 0}
    for ep in range(a.epochs):
        if ep == a.phase1_epochs:
            set_trainable(model, ['l1_encoder', 'l3_encoder'], True)
            for g in opt.param_groups:
                g['lr'] = a.lr2
        model.train()
        running = 0.0
        for step, batch in enumerate(train_ld):
            batch = {k: (v.to(a.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            loss, _ = compute_loss(model, batch, gender_weight=a.gender_weight, coral_weight=a.coral_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss)
        sched.step()
        m = evaluate(model, val_ld, a.device)
        mae = float(m['age_mae'] if isinstance(m, dict) else m[0])
        acc = float(m['sex_acc'] if isinstance(m, dict) else m[1])
        print('epoch %3d | train %.4f | val MAE %.4f | val GAcc %.4f | lr %.2e'
              % (ep + 1, running / max(1, step + 1), mae, acc, opt.param_groups[0]['lr']), flush=True)
        if mae < best['age']:
            best['age'], best_epoch['age'] = mae, ep + 1
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep + 1},
                       os.path.join(a.outdir, 'best_age.pt'))
        if acc > best['sex']:
            best['sex'], best_epoch['sex'] = acc, ep + 1
            torch.save({'model_state_dict': model.state_dict(), 'epoch': ep + 1},
                       os.path.join(a.outdir, 'best_sex.pt'))
        if ep + 1 >= a.min_epochs and (ep + 1 - min(best_epoch['age'], best_epoch['sex'])) >= a.patience:
            print('early stop at epoch', ep + 1, flush=True)
            break
    ck = torch.load(os.path.join(a.outdir, 'best_age.pt'), map_location=a.device)
    model.load_state_dict(ck['model_state_dict'])
    print('test:', evaluate(model, test_ld, a.device), flush=True)
    json.dump({'best_val_age_mae': best['age'], 'best_val_sex_acc': best['sex']},
              open(os.path.join(a.outdir, 'summary.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
