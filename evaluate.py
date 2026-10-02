"""Evaluate a trained HMA-Net checkpoint on one split of the manifest."""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hma_net'))
from hma_net import DentalAgeDataset, HMAAgePredictor, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint', required=True, help='checkpoint written by train.py (best_age.pt or best_sex.pt)')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=12)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    df = pd.read_csv(a.manifest)
    ds = DentalAgeDataset(df[df.split == a.split], img_size=a.img_size, training=False)
    ld = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers)

    model = HMAAgePredictor(img_size=a.img_size).to(a.device)
    ckpt = torch.load(a.checkpoint, map_location=a.device)
    state = ckpt.get('model_state_dict', ckpt)
    # Checkpoints from the reported runs also carry the projector of the L2 branch, which is
    # disabled in this release; it is unused at inference, so those tensors are ignored.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or missing:
        print('note: ignored %d tensor(s) not used by this release: %s'
              % (len(unexpected), sorted(unexpected)))
    model.eval()

    m = evaluate(model, ld, a.device)
    print('split %s | n %d | age MAE %.4f | sex accuracy %.4f'
          % (a.split, len(ds), m['mae'], m['gender_acc']))


if __name__ == '__main__':
    main()
