'''
jsons/
    aic21_train.json
    aic21_validation.json
videos/
    aic21_train/
        c001.imgs
        c001.gt
        c002.imgs
        ...
    aic21_validation/
        c006.imgs
        c006.gt
        c007.imgs
        ...
'''

import copy
import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
import os


def df2coco(df, video_dir):

    images = []
    annotations = []
    imgW, imgH = Image.open(video_dir / 'imgs' / '0001.jpg').size

    for (fid, group) in df.groupby('fid'):
        image_id = f'{video_dir.stem}-{fid}'
        images.append(
            {
                'id': image_id,
                'file_name': str(video_dir / 'imgs' / f'{fid:04d}.jpg'),
                'width': imgW,
                'height': imgH,
                'video': video_dir.stem,
            }
        )

        data = group[['x', 'y', 'w', 'h', 'tag']].values
        for (x, y, w, h, tag) in data.tolist():
            annotations.append(
                {
                    'id': f'{video_dir.stem}-{len(annotations)}',
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': (x, y, w, h),
                    'vis': 1.0,
                    'area': w * h,
                    'iscrowd': 0,
                }
            )
    return images, annotations


def normalize_ids(data):
    data = copy.deepcopy(data)

    image_ids = [ann['id'] for ann in data['images']]
    mapping = {x: i for i, x in enumerate(image_ids)}
    for img_ann in data['images']:
        img_ann['id'] = mapping[img_ann['id']]
    for box_ann in data['annotations']:
        box_ann['image_id'] = mapping[box_ann['image_id']]

    box_ids = [ann['id'] for ann in data['annotations']]
    mapping = {x: i for i, x in enumerate(box_ids)}
    for box_ann in data['annotations']:
        box_ann['id'] = mapping[box_ann['id']]

    assert len(set([ann['id'] for ann in data['images']])) == len(data['images'])
    assert len(set([ann['id'] for ann in data['annotations']])) == len(data['annotations'])
    assert len(set([ann['image_id'] for ann in data['annotations']])) == len(data['images'])

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aic21_train', type=Path, default='../mcmt/dataset/train')
    parser.add_argument('--aic21_validation', type=Path, default='../mcmt/dataset/validation')
    parser.add_argument('--jsons', type=Path, default='./jsons')
    parser.add_argument('--videos', type=Path, default='./videos/')
    args = parser.parse_args()

    assert args.aic21_train.exists()
    assert args.aic21_validation.exists()


    # Prepare videos
    args.videos.mkdir(parents=True)
    (args.videos / 'aic21_train').mkdir()
    (args.videos / 'aic21_validation').mkdir()

    for video_dir in args.aic21_train.glob('*/*'):
        video_name = video_dir.stem
        paths = sorted(list(video_dir.glob('imgs/*.jpg')))
        fids = [int(p.stem) for p in paths]
        df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
        df_imgs.to_csv(args.videos / 'aic21_train' / f'{video_name}.imgs', index=None)
        if (video_dir / 'gt' / 'gt.txt').exists():
            df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
            df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
            df_true.to_csv(args.videos / 'aic21_train' / f'{video_name}.gt', index=None)
            
    for video_dir in args.aic21_validation.glob('*/*'):
        video_name = video_dir.stem
        paths = sorted(list(video_dir.glob('imgs/*.jpg')))
        fids = [int(p.stem) for p in paths]
        df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
        df_imgs.to_csv(args.videos / 'aic21_validation' / f'{video_name}.imgs', index=None)
        if (video_dir / 'gt' / 'gt.txt').exists():
            df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
            df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
            df_true.to_csv(args.videos / 'aic21_validation' / f'{video_name}.gt', index=None)
