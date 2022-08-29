import torch
import argparse
import pandas as pd
from tqdm import tqdm
from model import Model
from pathlib import Path
from dataset import Video
from torch.utils.data import DataLoader
from torchvision.ops import box_convert


parser = argparse.ArgumentParser()
parser.add_argument('--videos', type=Path, default='videos/mot17')
parser.add_argument('--ckpt', type=Path, required=True)
parser.add_argument('--dets', type=Path, required=True)
parser.add_argument('--score_thresh', type=float, default=0.4)
parser.add_argument('--nms_thresh', type=float, default=0.7)
parser.add_argument('--num_proposals', type=int, default=300)
args = parser.parse_args()

assert args.videos.exists()
assert args.ckpt.exists()
args.dets.mkdir(parents=True, exist_ok=True)

model = Model.load_from_checkpoint(args.ckpt, num_proposals=args.num_proposals)
model = model.to('cuda:0')
model.eval()


for video_paths in sorted(list(args.videos.glob('*.imgs'))):
    video_name = video_paths.stem
    paths = pd.read_csv(video_paths)
    loader = DataLoader(
        Video(paths['fid'].tolist(), paths['path'].tolist()),
        batch_size=2,
        shuffle=False,
        num_workers=8,
        collate_fn=lambda x: x,
    )

    results = []
    for inps in tqdm(iter(loader), desc=video_name):
        with torch.no_grad():
            images = [item['image'].float() for item in inps]
            fpn_feats, sizes = model.extract_fpn_feats(images)
            logits, boxes, feats = model.detect(fpn_feats, sizes)

            alphas = model.unc_head(feats[-1])
            anchors = boxes[-2].detach()
            whs = anchors[..., 2:] - anchors[..., :2]  # [N, P, 2]
            sigmas = torch.exp(0.5 * alphas)  # [N, P, 4]
            sigmas = sigmas * torch.cat([whs, whs], dim=2)

            outs = model.postprocess(
                logits, boxes, sigmas, args.score_thresh, args.nms_thresh
            )

            for inp, (scores, labels, boxes, sigmas) in zip(inps, outs):
                boxes[:, 0::2] *= inp['rawW'] / inp['newW']
                boxes[:, 1::2] *= inp['rawH'] / inp['newH']
                boxes = box_convert(boxes.to('cpu'), 'xyxy', 'xywh')

                group = pd.DataFrame(boxes.numpy(), columns=['x', 'y', 'w', 'h'])
                group['tag'] = -1
                group['fid'] = inp['fid']
                group['s'] = scores.to('cpu').tolist()
                group[['ul', 'ut', 'ur', 'ub']] = sigmas.to('cpu').numpy()
                results.append(group)

    df_dets = pd.concat(results, ignore_index=True)
    df_dets.to_csv(
        args.dets / f'{video_name}.csv',
        index=None,
        columns=['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub'],
    )
