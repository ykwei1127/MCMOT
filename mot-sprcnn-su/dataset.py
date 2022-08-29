import json
import torch
import numpy as np
import imgaug.augmenters as iaa
from PIL import Image
from detectron2.structures import Instances
from torch.utils.data import Dataset
from collections import defaultdict


class Video(Dataset):
    def __init__(self, fids, paths):
        super().__init__()
        assert len(fids) == len(paths)
        self.fids = fids
        self.paths = paths

        self.aug = iaa.Sequential(
            [
                iaa.Resize(
                    {
                        # 'shorter-side': [800],
                        'shorter-side': [1080],
                        'longer-side': 'keep-aspect-ratio',
                    }
                ),
            ]
        )

    def __len__(self):
        return len(self.fids)

    def __getitem__(self, idx):
        fid = self.fids[idx]
        path = self.paths[idx]

        image = np.array(Image.open(path))
        rawH, rawW, _ = image.shape

        (image,) = self.aug.augment(images=[image])
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        _, newH, newW = image.shape

        return {
            'fid': fid,
            'image': image,
            'newW': newW,
            'newH': newH,
            'rawW': rawW,
            'rawH': rawH,
        }


class TrainData(Dataset):
    def __init__(self, json_path, mode='infer'):
        with open(json_path) as f:
            data = json.load(f)
        self.meta = data['images']
        self.anns = defaultdict(list)
        for ann in data['annotations']:
            self.anns[ann['image_id']].append(ann)
        self.image_ids = list(self.anns.keys())

        if mode != 'train':
            self.aug = iaa.Resize(
                {
                    'shorter-side': [800],
                    'longer-side': 'keep-aspect-ratio',
                }
            )
        else:
            self.aug = iaa.Sequential(
                [
                    iaa.Fliplr(0.5),
                    iaa.Resize(
                        {
                            'shorter-side': [640, 672, 704, 736, 768, 800],
                            'longer-side': 'keep-aspect-ratio',
                        }
                    ),
                    iaa.CropToFixedSize(1500, 1500),
                    iaa.Sometimes(
                        0.3,
                        [iaa.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, +0.1))],
                    ),
                    iaa.Sometimes(
                        0.3,
                        [
                            iaa.AddToSaturation((-30, +30)),
                            iaa.AddToBrightness((-30, +30)),
                            iaa.LinearContrast((0.8, 1.2)),
                        ],
                    ),
                    iaa.Sometimes(
                        0.1, [iaa.OneOf([iaa.MotionBlur(), iaa.GaussianBlur()])]
                    ),
                ]
            )

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        meta = self.meta[image_id]
        anns = self.anns[image_id]

        image = np.array(Image.open(meta['file_name']))
        boxes = [ann['bbox'] for ann in anns]
        boxes = np.float32(boxes)
        boxes[:, 2:] += boxes[:, :2]
        rawH, rawW, _ = image.shape

        (image,), (boxes,) = self.aug.augment(images=[image], bounding_boxes=[boxes])

        image = torch.from_numpy(image.copy()).permute(2, 0, 1)

        H, W, _ = image.shape
        insts = Instances((H, W))
        insts.gt_boxes = torch.from_numpy(boxes).float()
        insts.gt_classes = torch.zeros(len(insts)).long()

        return {
            'image_id': image_id,
            'image': image,
            'insts': insts,
            'rawW': rawW,
            'rawH': rawH,
        }
