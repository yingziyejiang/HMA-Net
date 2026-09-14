

import os, sys, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd, cv2
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
import torchvision.models as models

ROOT = os.environ.get("HMA_DATA_ROOT", ".")

LIGHT_FEAT_DIMS = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}

def _build_backbone(name):
    bb = getattr(models, name)(weights='DEFAULT')
    return nn.Sequential(*list(bb.children())[:-1]), LIGHT_FEAT_DIMS[name]

class DentalAgeDataset(Dataset):
    def __init__(self, df, img_size=256, tooth_size=64, max_teeth=52, training=False,
                 jitter_scale=0.12, jitter_shift=0.15, jitter_angle=12.0, tooth_dropout=0.2, color_jitter=True,
                 context_margin=0.6):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.tooth_size = tooth_size
        self.max_teeth = max_teeth
        self.training = training
        self.jitter_scale = jitter_scale
        self.jitter_shift = jitter_shift
        self.jitter_angle = jitter_angle
        self.tooth_dropout = tooth_dropout
        self.color_jitter = color_jitter
        self.context_margin = context_margin

        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def _load_img(self, path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.float32), (1, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.img_mean) / self.img_std
        return img.transpose(2, 0, 1), (w, h)

    def _load_labels(self, path, use_obb5=False):
        boxes, fdi_ids = [], []
        if not os.path.exists(path):
            return np.array(boxes), np.array(fdi_ids)
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 6:

                    fdi = int(parts[0])
                    cx = float(parts[1])
                    cy = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    angle = float(parts[5])
                    boxes.append([cx, cy, w, h, angle])
                    fdi_ids.append(fdi)
                elif len(parts) >= 9:

                    fdi = int(parts[0])
                    pts = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
                    if use_obb5:

                        rect = cv2.minAreaRect(pts)
                        cx, cy = rect[0]
                        w, h = rect[1]
                        angle = rect[2]
                    else:

                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        cx = (x_min + x_max) / 2.0
                        cy = (y_min + y_max) / 2.0
                        w = x_max - x_min
                        h = y_max - y_min
                        angle = 0.0
                    boxes.append([cx, cy, w, h, angle])
                    fdi_ids.append(fdi)
        return np.array(boxes, dtype=np.float32), np.array(fdi_ids, dtype=np.int64)

    def _crop_tooth(self, img_full, box, orig_w, orig_h):
        cx, cy, w, h, angle = box

        if self.training:

            w = w * (1.0 + np.random.uniform(-self.jitter_scale, self.jitter_scale))
            h = h * (1.0 + np.random.uniform(-self.jitter_scale, self.jitter_scale))

            size = max(w, h)
            cx = cx + np.random.uniform(-self.jitter_shift, self.jitter_shift) * size / orig_w
            cy = cy + np.random.uniform(-self.jitter_shift, self.jitter_shift) * size / orig_h

            angle = angle + np.random.uniform(-self.jitter_angle, self.jitter_angle)

        cx_img = cx * orig_w
        cy_img = cy * orig_h
        w_img = w * orig_w
        h_img = h * orig_h

        size = max(w_img, h_img) * 1.4 * (1.0 + self.context_margin)
        if size < 20:
            size = 20

        M = cv2.getRotationMatrix2D((cx_img, cy_img), angle, 1.0)
        M[0, 2] += size / 2 - cx_img
        M[1, 2] += size / 2 - cy_img
        try:
            crop = cv2.warpAffine(img_full, M, (int(size), int(size)),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        except:
            return np.zeros((self.tooth_size, self.tooth_size, 3), dtype=np.float32)
        crop = cv2.resize(crop, (self.tooth_size, self.tooth_size))

        if self.training and self.color_jitter:
            crop = crop.astype(np.float32)

            brightness = 1.0 + np.random.uniform(-0.15, 0.15)
            crop = crop * brightness

            contrast = 1.0 + np.random.uniform(-0.20, 0.20)
            mean = crop.mean()
            crop = (crop - mean) * contrast + mean
            crop = np.clip(crop, 0, 255)

        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.img_mean) / self.img_std
        return crop.transpose(2, 0, 1)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(ROOT, row['img_path'])
        label_path = os.path.join(ROOT, row['label_path'])
        age = float(row['age'])
        gender = int(row['gender'])

        img_tensor, (orig_w, orig_h) = self._load_img(img_path)
        boxes, fdi_ids = self._load_labels(label_path)

        img_full = cv2.imread(img_path)
        if img_full is not None:
            img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
        else:
            img_full = np.zeros((max(orig_h, 1), max(orig_w, 1), 3), dtype=np.uint8)
        img_h, img_w = img_full.shape[:2]

        teeth = np.zeros((self.max_teeth, 3, self.tooth_size, self.tooth_size), dtype=np.float32)
        pos_bbox = np.zeros((self.max_teeth, 4), dtype=np.float32)
        pos_fdi = np.zeros(self.max_teeth, dtype=np.int64)
        mask = np.zeros(self.max_teeth, dtype=np.float32)

        n_valid = 0
        for i, (box, fdi) in enumerate(zip(boxes, fdi_ids)):
            if i >= self.max_teeth:
                break
            teeth[i] = self._crop_tooth(img_full, box, orig_w, orig_h)
            pos_bbox[i] = box[:4]
            pos_fdi[i] = fdi
            mask[i] = 1.0
            n_valid += 1

        if self.training and self.tooth_dropout > 0 and n_valid > 0:
            valid_idx = np.where(mask[:n_valid] == 1.0)[0]
            n_drop = max(1, int(len(valid_idx) * self.tooth_dropout))
            drop_idx = np.random.choice(valid_idx, n_drop, replace=False)
            teeth[drop_idx] = 0.0

        return {
            'img': torch.FloatTensor(img_tensor),
            'teeth': torch.FloatTensor(teeth),
            'pos_bbox': torch.FloatTensor(pos_bbox),
            'fdi': torch.LongTensor(pos_fdi),
            'mask': torch.FloatTensor(mask),
            'age': torch.FloatTensor([age]),
            'gender': torch.FloatTensor([gender]),
        }

class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim=256, backbone_type='resnet50', img_size=256):
        super().__init__()
        self.features, self.feat_dim = _build_backbone(backbone_type)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(self.feat_dim, embed_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim)
        )
        self.img_size = img_size

    def forward(self, x):
        feats = self.features(x)
        feats = self.pool(feats).flatten(1)
        return self.proj(feats)

class ToothEncoder(nn.Module):
    def __init__(self, embed_dim=256, backbone_type='resnet34', tooth_size=64):
        super().__init__()
        self.tooth_size = tooth_size
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bb = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if backbone_type == 'resnet34':
            pass
        elif backbone_type == 'resnet18':
            bb = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.bb = nn.Sequential(conv1, bb.bn1, bb.relu, bb.maxpool,
                                bb.layer1, bb.layer2, bb.layer3, bb.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        bb_dim = 512
        self.pos_proj = nn.Linear(4, 32)
        self.fdi_embed = nn.Embedding(49, 16)
        self.proj = nn.Sequential(
            nn.Linear(bb_dim + 32 + 16, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, teeth, pos_bbox, fdi):
        feats = self.pool(self.bb(teeth)).flatten(1)
        pos = self.pos_proj(pos_bbox)
        fdi_emb = self.fdi_embed(fdi.clamp(0, 48))
        return self.proj(torch.cat([feats, pos, fdi_emb], dim=-1))

class SpatialGAT(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, topology='knn', top_k=4,
):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.topology = topology
        self.top_k = top_k

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_attn = nn.Linear(4, num_heads)

    def forward(self, tooth_feats, pos_bbox, mask):
        B, N, D = tooth_feats.shape

        q = self.q_proj(tooth_feats).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(tooth_feats).view(B, N, self.num_heads, self.head_dim)
        attn = torch.einsum('bnhd,bmhd->bhnm', q, k) * self.scale

        pos_score = self.pos_attn(pos_bbox).permute(0, 2, 1).unsqueeze(2)
        attn = attn + pos_score

        if self.topology == 'knn':
            topk = min(self.top_k, N)
            _, topk_idx = attn.topk(topk, dim=-1)
            knn_mask = torch.zeros_like(attn).scatter_(-1, topk_idx, 1.0)
            attn = attn.masked_fill(knn_mask == 0, -1e9)

        mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(1)
        attn = attn.masked_fill(mask_2d.unsqueeze(1) == 0, -1e9)
        attn = F.softmax(attn, dim=-1)

        v = self.v_proj(tooth_feats).view(B, N, self.num_heads, self.head_dim)
        out = torch.einsum('bhnm,bmhd->bnhd', attn, v)
        out = out.reshape(B, N, D)
        return self.out_proj(out)

class HMAAgePredictor(nn.Module):
    def __init__(self, embed_dim=256, num_age_bins=16, use_coral=True,
                 l1_backbone='resnet50', l3_backbone='resnet34', img_size=256):
        super().__init__()
        self.use_coral = use_coral
        self.embed_dim = embed_dim
        self.use_gender_l1 = True
        self.use_task_decoupled = True

        self.l1_encoder = GlobalEncoder(embed_dim, backbone_type=l1_backbone, img_size=img_size)
        self.l3_encoder = ToothEncoder(embed_dim, backbone_type=l3_backbone)
        self.l3_aggr = SpatialGAT(embed_dim, num_heads=4, topology='knn', top_k=4)

        l1_dim = l3_dim = embed_dim

        self.l3_adapter = nn.Sequential(
            nn.Linear(l3_dim, l3_dim), nn.LayerNorm(l3_dim), nn.GELU(),
            nn.Linear(l3_dim, l3_dim),
        )

        gender_l1_dim = embed_dim // 2
        self.gender_l1_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(embed_dim, gender_l1_dim), nn.GELU(),
        )
        self.gender_l1_dim = gender_l1_dim

        age_fused_dim = l1_dim + l3_dim
        gender_fused_dim = gender_l1_dim + l3_dim
        self.film_age_gen = nn.Sequential(
            nn.Linear(l3_dim, l3_dim), nn.GELU(),
            nn.Linear(l3_dim, age_fused_dim * 2),
        )
        self.film_gender_gen = nn.Sequential(
            nn.Linear(l3_dim, l3_dim), nn.GELU(),
            nn.Linear(l3_dim, gender_fused_dim * 2),
        )
        self.fusion_type = 'film'

        age_mlp_in = l1_dim + l3_dim
        self.age_fusion_mlp = nn.Sequential(
            nn.Linear(age_mlp_in, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
        )

        gender_mlp_in = gender_l1_dim + l3_dim
        self.gender_fusion_mlp = nn.Sequential(
            nn.Linear(gender_mlp_in, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1),
        )

        self.gender_embed = nn.Embedding(2, 16)

        self.age_head = nn.Sequential(
            nn.Linear(128 + 16, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

        gender_head_in = 64 + (gender_l1_dim if self.use_gender_l1 else 0)
        self.gender_head = nn.Sequential(
            nn.Linear(gender_head_in, 32), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

        if use_coral:
            self.coral_head = nn.Linear(128, num_age_bins - 1)

    def forward(self, batch):
        img = batch['img']
        teeth = batch['teeth']
        pos_bbox = batch['pos_bbox']
        fdi = batch['fdi']
        mask = batch['mask']
        gender = batch['gender'].squeeze(-1)
        B, N = teeth.shape[:2]

        l1_feat = self.l1_encoder(img)

        teeth_flat = teeth.view(B * N, 3, 64, 64)
        bbox_flat = pos_bbox.view(B * N, 4)
        fdi_flat = fdi.view(B * N)
        tooth_feats = self.l3_encoder(teeth_flat, bbox_flat, fdi_flat)
        tooth_feats = tooth_feats.view(B, N, -1)
        if self.l3_aggr is not None:
            tooth_seq = self.l3_aggr(tooth_feats, pos_bbox, mask)
        else:
            tooth_seq = tooth_feats
        l3_feat_raw = (tooth_seq * mask.unsqueeze(-1)).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)

        l3_feat = self.l3_adapter(l3_feat_raw)

        if self.use_gender_l1:
            l1_gender = self.gender_l1_proj(l1_feat)
        else:
            l1_gender = None

        age_concat = torch.cat([l1_feat, l3_feat], dim=-1)
        film_out = self.film_age_gen(l3_feat)
        half = film_out.shape[-1] // 2
        age_gated = age_concat * film_out[:, :half] + film_out[:, half:]
        age_feat = self.age_fusion_mlp(age_gated)

        gender_l1_input = l1_gender if l1_gender is not None else l1_feat
        gender_concat = torch.cat([gender_l1_input, l3_feat], dim=-1)
        film_out = self.film_gender_gen(l3_feat)
        half = film_out.shape[-1] // 2
        gender_gated = gender_concat * film_out[:, :half] + film_out[:, half:]
        gender_feat = self.gender_fusion_mlp(gender_gated)

        gen_emb = self.gender_embed(gender.long())
        age_input = torch.cat([age_feat, gen_emb], dim=-1)
        age_pred = self.age_head(age_input)

        if self.use_gender_l1:
            gender_input = torch.cat([gender_feat, l1_gender], dim=-1)
        else:
            gender_input = gender_feat
        gender_logit = self.gender_head(gender_input)

        coral_logits = self.coral_head(age_feat) if self.use_coral else None

        return (age_pred, gender_logit, coral_logits)

def focal_bce_loss(logits, targets, gamma=2.0, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = torch.exp(-bce)

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    focal_weight = alpha_t * (1 - pt) ** gamma
    return (focal_weight * bce).mean()

def coral_loss(logits, ages, num_bins=16, age_min=3.0, age_max=18.0):
    bin_width = (age_max - age_min) / num_bins
    levels = torch.clamp(((ages - age_min) / bin_width).long(), 0, num_bins - 1)
    levels = levels.squeeze(-1)
    targets = torch.zeros(len(ages), num_bins - 1, device=ages.device)
    for k in range(num_bins - 1):
        targets[:, k] = (levels > k).float()
    return F.binary_cross_entropy_with_logits(logits, targets)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_ages, all_genders, all_gender_preds = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)
            age_pred, gender_logit = out[0], out[1]
            all_preds.append(age_pred.cpu().numpy())
            all_ages.append(batch['age'].cpu().numpy())
            all_genders.append(batch['gender'].cpu().numpy())
            all_gender_preds.append(torch.sigmoid(gender_logit).cpu().numpy())
    preds = np.concatenate(all_preds).flatten()
    ages = np.concatenate(all_ages).flatten()
    genders = np.concatenate(all_genders).flatten()
    gender_preds = (np.concatenate(all_gender_preds).flatten() > 0.5).astype(np.float32)
    mae = np.abs(preds - ages).mean()
    rmse = np.sqrt(((preds - ages) ** 2).mean())
    medae = np.median(np.abs(preds - ages))
    gender_acc = (gender_preds == genders).mean()
    male_acc = (gender_preds[genders == 1] == 1).mean() if (genders == 1).sum() > 0 else 0.0
    female_acc = (gender_preds[genders == 0] == 0).mean() if (genders == 0).sum() > 0 else 0.0
    return {'mae': float(mae), 'rmse': float(rmse), 'medae': float(medae),
            'gender_acc': float(gender_acc), 'male_acc': float(male_acc),
            'female_acc': float(female_acc),
            'preds': preds.tolist(), 'ages': ages.tolist(),
            'gender_preds': gender_preds.tolist(), 'genders': genders.tolist()}

def set_trainable(model, patterns, trainable=True):
    for name, param in model.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = trainable

def compute_loss(model, batch, loss_type='fixed_weight',
                    use_focal=True, focal_gamma=2.0, focal_alpha=0.5,
                    gender_weight=3.0, coral_weight=0.5):
    age_pred, gender_logit, coral_logits = model(batch)

    age_mse = F.mse_loss(age_pred, batch['age'])
    if use_focal:
        gender_loss = focal_bce_loss(gender_logit, batch['gender'], gamma=focal_gamma, alpha=focal_alpha)
    else:
        gender_loss = F.binary_cross_entropy_with_logits(gender_logit, batch['gender'])
    coral_l = coral_loss(coral_logits, batch['age']) if coral_logits is not None else None

    loss = age_mse + gender_weight * gender_loss
    if coral_l is not None:
        loss = loss + coral_weight * coral_l

    return loss, age_mse.item(), gender_loss.item(), coral_l.item() if coral_l is not None else 0.0

class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        df = dataset.df
        self.male_idx = df[df['gender'] == 1].index.tolist()
        self.female_idx = df[df['gender'] == 0].index.tolist()
        self.n_batches = (len(self.male_idx) + len(self.female_idx)) // batch_size

    def __iter__(self):
        half = self.batch_size // 2
        male_pool = self.male_idx.copy()
        female_pool = self.female_idx.copy()
        if self.shuffle:
            np.random.shuffle(male_pool)
            np.random.shuffle(female_pool)

        batches = []

        for b in range(self.n_batches):
            batch = []

            m_start = (b * half) % len(male_pool)
            for i in range(half):
                batch.append(male_pool[(m_start + i) % len(male_pool)])

            f_start = (b * half) % len(female_pool)
            for i in range(half):
                batch.append(female_pool[(f_start + i) % len(female_pool)])
            if self.shuffle:
                np.random.shuffle(batch)
            batches.append(batch)

        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return self.n_batches
