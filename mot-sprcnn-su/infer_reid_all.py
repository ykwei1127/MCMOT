import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision.ops import box_convert
from torchvision.transforms import functional as TF

import sys
from reid.config import cfg
from reid.reid_inference.reid_model import build_reid_model, build_reid_model_2, build_reid_model_3
import numpy as np
from sklearn import preprocessing
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--videos', type=Path, default='videos/aic21')
parser.add_argument('--dets', type=Path, required=True)
parser.add_argument('--flip_feature', action='store_true')
args = parser.parse_args()

assert args.videos.exists()
assert args.dets.exists()

# print(cfg)
reid_model, reid_cfg = build_reid_model(cfg)
reid_model = reid_model.to('cuda')
reid_model.eval()

reid_model_2, reid_cfg = build_reid_model_2(cfg)
reid_model_2 = reid_model_2.to('cuda')
reid_model_2.eval()

reid_model_3, reid_cfg = build_reid_model_3(cfg)
reid_model_3 = reid_model_3.to('cuda')
reid_model_3.eval()


for video_paths in sorted(list(args.videos.glob('*.imgs'))):
    video_name = video_paths.stem
    paths = {ann.fid: ann.path for ann in pd.read_csv(video_paths).itertuples()}
    dets = pd.read_csv(args.dets / f'{video_name}.csv')
    embs = torch.zeros(len(dets), 2048)
    check = torch.zeros(len(dets))

    for fid, group in tqdm(dets.groupby('fid'), desc=video_name):
        image = Image.open(paths[fid])

        patches = []
        for ann in group.itertuples():
            patch = image.crop((ann.x, ann.y, ann.x + ann.w, ann.y + ann.h))
            patch = patch.resize(tuple(reid_cfg.INPUT.SIZE_TEST))
            # patch = patch.resize((384, 384))
            patch = TF.to_tensor(patch)
            patches.append(patch)
        patches = torch.stack(patches)
        patches = TF.normalize(patches, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.no_grad():
            if args.flip_feature:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(patches.size(3) - 1, -1, -1).long()
                        patches = patches.index_select(3, inv_idx)
                        feat1 = reid_model(patches.to('cuda')).to('cpu').detach().numpy()
                    else:
                        feat2 = reid_model(patches.to('cuda')).to('cpu').detach().numpy()
                extract = feat2 + feat1
            else:
                extract = reid_model(patches.to('cuda')).to('cpu').detach().numpy()
            patch_feature_array_1 = np.array(extract)
            patch_feature_array_1 = preprocessing.normalize(patch_feature_array_1, norm='l2', axis=1)

            if args.flip_feature:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(patches.size(3) - 1, -1, -1).long()
                        patches = patches.index_select(3, inv_idx)
                        feat1 = reid_model_2(patches.to('cuda')).to('cpu').detach().numpy()
                    else:
                        feat2 = reid_model_2(patches.to('cuda')).to('cpu').detach().numpy()
                extract = feat2 + feat1
            else:
                extract = reid_model_2(patches.to('cuda')).to('cpu').detach().numpy()
            patch_feature_array_2 = np.array(extract)
            patch_feature_array_2 = preprocessing.normalize(patch_feature_array_2, norm='l2', axis=1)

            if args.flip_feature:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(patches.size(3) - 1, -1, -1).long()
                        patches = patches.index_select(3, inv_idx)
                        feat1 = reid_model_3(patches.to('cuda')).to('cpu').detach().numpy()
                    else:
                        feat2 = reid_model_3(patches.to('cuda')).to('cpu').detach().numpy()
                extract = feat2 + feat1
            else:
                extract = reid_model_3(patches.to('cuda')).to('cpu').detach().numpy()
            patch_feature_array_3 = np.array(extract)
            patch_feature_array_3 = preprocessing.normalize(patch_feature_array_3, norm='l2', axis=1)

            patch_feature_array = np.array([patch_feature_array_1, patch_feature_array_2, patch_feature_array_3])

            patch_feature_mean = np.mean(patch_feature_array, axis=0)
  
            embs[group.index.to_numpy()] = torch.tensor(patch_feature_mean)
        check[group.index.to_numpy()] += 1
    
    assert torch.all(check == 1)
    torch.save(embs, args.dets / f'{video_name}.emb')
