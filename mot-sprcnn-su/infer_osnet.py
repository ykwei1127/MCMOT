import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torchvision.ops import box_convert
from torchvision.transforms import functional as TF

from torchreid.models import build_model
from torchreid.utils import load_pretrained_weights


parser = argparse.ArgumentParser()
parser.add_argument('--videos', type=Path, default='videos/mot17')
parser.add_argument('--dets', type=Path, required=True)
args = parser.parse_args()

assert args.videos.exists()
assert args.dets.exists()

reid_model = build_model('osnet_x1_0', 1)
load_pretrained_weights(reid_model, 'weights/osnet_x1_0_MS_D_C.pth')
reid_model = reid_model.to('cuda')
reid_model.eval()


for video_paths in sorted(list(args.videos.glob('*.imgs'))):
    video_name = video_paths.stem
    paths = {ann.fid: ann.path for ann in pd.read_csv(video_paths).itertuples()}
    dets = pd.read_csv(args.dets / f'{video_name}.csv')
    embs = torch.zeros(len(dets), 512)
    check = torch.zeros(len(dets))

    for fid, group in tqdm(dets.groupby('fid'), desc=video_name):
        image = Image.open(paths[fid])

        patches = []
        for ann in group.itertuples():
            patch = image.crop((ann.x, ann.y, ann.x + ann.w, ann.y + ann.h))
            patch = patch.resize((128, 256))
            patch = TF.to_tensor(patch)
            patches.append(patch)
        patches = torch.stack(patches)
        patches = TF.normalize(patches, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        with torch.no_grad():
            embs[group.index.to_numpy()] = reid_model(patches.to('cuda')).to('cpu')
        check[group.index.to_numpy()] += 1
    
    assert torch.all(check == 1)
    torch.save(embs, args.dets / f'{video_name}.emb')
