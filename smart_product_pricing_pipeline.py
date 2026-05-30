"""
Smart Product Pricing — starter pipeline
Filename: smart_product_pricing_pipeline.py
Purpose: Full end-to-end starter pipeline for ML Challenge 2025 Smart Product Pricing.
Outputs: `test_out.csv` matching required format: [sample_id, price]

Notes:
- This is a robust starter that combines text features (TF-IDF + IPQ extraction + basic text cleaning),
  optional transformer text embeddings (commented), image features via pretrained ResNet, and a LightGBM model.
- It respects the "no external price lookup" rule: uses only the provided CSVs and product images from the given image_link column.
- Adjust file paths, hyperparameters, and resource settings as needed.
- The script is intended as a single-file starter for experimentation and extension; convert to a notebook if you prefer.

Requirements (pip install):
  pandas, numpy, scikit-learn, lightgbm, tqdm, pillow, torchvision, torch, regex
  (optional: transformers, sentence-transformers for better text embeddings)

Run example:
  python smart_product_pricing_pipeline.py --train dataset/train.csv --test dataset/test.csv --out test_out.csv --img-dir images

"""

import os
import re
import argparse
import gc
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb

# Optional image / torch imports
try:
    import torch
    import torchvision.transforms as T
    from torchvision import models
    from PIL import Image
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# --------------------------- Helpers ---------------------------

def extract_ipq(text):
    """Extract Item Pack Quantity (IPQ)-like numbers from the catalog_content.
    Returns integer or np.nan if not found.
    Examples handled: 'Pack of 2', '2 Pack', '500 ml', '250g', '10 Count', 'Qty: 6', '6pcs', '6 pcs'
    """
    if not isinstance(text, str):
        return np.nan
    text_low = text.lower()
    # common patterns: 'pack of 2', '2 pack', '2 pcs', 'qty: 3', '3 count'
    patterns = [r'pack of\s*(\d+)', r'(\d+)\s*pack\b', r'(\d+)\s*pcs?\b',
                r'qty[:\s]*(\d+)', r'(\d+)\s*count\b', r'(\d+)\s*pieces\b',
                r'(\d+)\s*x\b', r'\b(\d+)\s*ml\b', r'\b(\d+)\s*g\b']
    for p in patterns:
        m = re.search(p, text_low)
        if m:
            try:
                return int(m.group(1))
            except:
                continue
    # fallback: any standalone number under 100
    nums = re.findall(r'\b(\d{1,3})\b', text_low)
    for n in nums:
        v = int(n)
        if 1 <= v <= 100:
            return v
    return np.nan


def simple_text_clean(text):
    if not isinstance(text, str):
        return ''
    # lower, remove urls, punctuation (keep alphanum), collapse spaces
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# --------------------------- Image feature extractor ---------------------------

class ImageFeatureExtractor:
    def __init__(self, device='cpu'):
        if not TORCH_AVAILABLE:
            raise RuntimeError('torch/torchvision not available - install torch and torchvision to extract image features')
        self.device = torch.device(device)
        # use a pretrained ResNet50 here; it's a good balance of speed/quality
        self.model = models.resnet50(pretrained=True)
        # remove classification head
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(self.device)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, pil_image):
        with torch.no_grad():
            img = self.transform(pil_image).unsqueeze(0).to(self.device)
            feat = self.model(img)
            return feat.squeeze(0).cpu().numpy()


# --------------------------- Main pipeline ---------------------------

def build_basic_text_features(df, text_col='catalog_content', n_tfidf=5000, svd_dim=128):
    print('Building text features...')
    df['clean_text'] = df[text_col].fillna('').astype(str).apply(simple_text_clean)
    df['ipq'] = df[text_col].apply(extract_ipq)

    # TF-IDF + SVD
    tfidf = TfidfVectorizer(max_features=n_tfidf, ngram_range=(1,2))
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    print('TF-IDF shape:', X_tfidf.shape)
    svd = TruncatedSVD(n_components=min(svd_dim, X_tfidf.shape[1]-1), random_state=42)
    X_svd = svd.fit_transform(X_tfidf)
    print('SVD reduced to:', X_svd.shape)

    text_feat_cols = [f'text_svd_{i}' for i in range(X_svd.shape[1])]
    df_text_feats = pd.DataFrame(X_svd, columns=text_feat_cols, index=df.index)

    # Basic length / token features
    df['char_len'] = df['clean_text'].str.len()
    df['word_len'] = df['clean_text'].str.split().apply(len)

    df_out = pd.concat([df[['sample_id', 'ipq', 'char_len', 'word_len']], df_text_feats], axis=1)
    return df_out, tfidf, svd


def build_image_features(df, image_dir, batch_limit=None, device='cpu'):
    if not TORCH_AVAILABLE:
        print('Torch not available, skipping image features')
        return pd.DataFrame(index=df.index)

    print('Building image features...')
    img_ext = ['.jpg', '.jpeg', '.png']
    extractor = ImageFeatureExtractor(device=device)

    feats = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        sid = row['sample_id']
        # expected image path in image_dir with filename = sample_id + ext OR try to download? assume already downloaded
        found = False
        for ext in img_ext:
            p = Path(image_dir) / f"{sid}{ext}"
            if p.exists():
                try:
                    pil = Image.open(p).convert('RGB')
                    feats[sid] = extractor.extract(pil)
                    found = True
                except Exception:
                    found = False
                    continue
        if not found:
            # try from image_link if present
            link = row.get('image_link', None)
            if isinstance(link, str) and link.strip():
                try:
                    # attempt to open via PIL directly from link (requires internet) - user may prefer to pre-download images
                    pil = Image.open(link).convert('RGB')
                    feats[sid] = extractor.extract(pil)
                except Exception:
                    feats[sid] = np.zeros(2048, dtype=np.float32)
            else:
                feats[sid] = np.zeros(2048, dtype=np.float32)

        if batch_limit and len(feats) >= batch_limit:
            break

    img_feat_df = pd.DataFrame.from_dict(feats, orient='index')
    img_feat_df.index.name = 'sample_id'
    img_feat_df = img_feat_df.rename_axis(None).reset_index()
    return img_feat_df


def train_lightgbm(X, y, X_test, params=None, n_splits=5, seed=42):
    print('Training LightGBM...')
    if params is None:
        params = {
            'objective':'regression',
            'metric':'mae',
            'verbosity':-1,
            'boosting_type':'gbdt',
            'learning_rate':0.05,
            'num_leaves':127,
            'feature_fraction':0.8,
            'bagging_freq':1,
            'bagging_fraction':0.8,
            'seed':seed
        }
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    feature_importance_df = []

    for fold, (tr_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold', fold+1)
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        model = lgb.train(params, dtrain, valid_sets=[dval], num_boost_round=10000,
                          early_stopping_rounds=100, verbose_eval=200)
        oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

    mae = mean_absolute_error(y, oof)
    print('OOF MAE:', mae)
    return preds, oof


def prepare_and_train(train_path, test_path, image_dir=None, out_path='test_out.csv'):
    print('Loading data...')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # safety: ensure sample_id present
    assert 'sample_id' in train.columns and 'sample_id' in test.columns

    # build text features (fit on combined to ensure same columns)
    all_df = pd.concat([train[['sample_id', 'catalog_content']], test[['sample_id', 'catalog_content']]], axis=0).reset_index(drop=True)
    text_feats, tfidf, svd = build_basic_text_features(all_df, text_col='catalog_content', n_tfidf=5000, svd_dim=128)

    # split back
    n_train = len(train)
    train_text = text_feats.iloc[:n_train].set_index('sample_id')
    test_text = text_feats.iloc[n_train:].set_index('sample_id')

    # Merge with any other columns
    train_feats = train_text.copy()
    test_feats = test_text.copy()

    # Add image features if available
    if image_dir is not None and os.path.exists(image_dir):
        # image_dir should contain files named <sample_id>.jpg/.png etc OR you should download images before running this
        img_all = pd.concat([train[['sample_id','image_link']], test[['sample_id','image_link']]], axis=0).reset_index(drop=True)
        img_feats = build_image_features(img_all, image_dir=image_dir)
        if not img_feats.empty:
            img_feats = img_feats.set_index('sample_id')
            train_img = img_feats.reindex(train_feats.index).fillna(0)
            test_img = img_feats.reindex(test_feats.index).fillna(0)
            # reduce dimensionality for speed
            scaler = StandardScaler()
            train_img = pd.DataFrame(scaler.fit_transform(train_img), index=train_img.index, columns=[f'img_{i}' for i in range(train_img.shape[1])])
            test_img = pd.DataFrame(scaler.transform(test_img), index=test_img.index, columns=[f'img_{i}' for i in range(test_img.shape[1])])
            train_feats = pd.concat([train_feats, train_img], axis=1)
            test_feats = pd.concat([test_feats, test_img], axis=1)

    # Final assemble: align indices & merge target
    train_feats = train_feats.reset_index().set_index('sample_id')
    test_feats = test_feats.reset_index().set_index('sample_id')

    # add target
    y = train.set_index('sample_id')['price'].loc[train_feats.index]

    # Fill NA
    train_feats = train_feats.fillna(0)
    test_feats = test_feats.fillna(0)

    # Optionally scale numeric
    # train_feats = pd.DataFrame(StandardScaler().fit_transform(train_feats), index=train_feats.index, columns=train_feats.columns)
    # test_feats = pd.DataFrame(StandardScaler().transform(test_feats), index=test_feats.index, columns=test_feats.columns)

    # Train model
    preds, _ = train_lightgbm(train_feats, y, test_feats)

    # Ensure positive float predictions
    preds = np.maximum(preds, 0.01)

    submission = pd.DataFrame({'sample_id': test_feats.index, 'price': preds})
    submission = submission.reset_index(drop=True)
    submission.to_csv(out_path, index=False)
    print('Saved submission to', out_path)


# --------------------------- CLI ---------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--out', default='test_out.csv')
    parser.add_argument('--img-dir', default=None, help='Directory where images are downloaded as <sample_id>.jpg')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    prepare_and_train(args.train, args.test, image_dir=args.img_dir, out_path=args.out)
