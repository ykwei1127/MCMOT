import copy
import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
import os

def load_odgt(odgt_path):
    img_dir = odgt_path.parent / 'Images'
    img_anns = []
    box_anns = []
    with open(odgt_path) as f:
        data = [json.loads(line) for line in f]
    for datum in data:
        img_id = datum['ID']
        img_path = str(img_dir / img_id) + '.jpg'
        imgW, imgH = Image.open(img_path).size

        fboxes = []
        vboxes = []
        for inst in datum['gtboxes']:
            fx, fy, fw, fh = inst['fbox']
            vx, vy, vw, vh = inst['vbox']
            if inst['tag'] != 'person':
                continue
            fboxes.append((fx, fy, fw, fh))
            vboxes.append((vx, vy, vw, vh))

        if len(fboxes) > 1:
            img_anns.append(
                {
                    'id': img_id,
                    'file_name': img_path,
                    'width': imgW,
                    'height': imgH,
                    'video': 'crowdhuman',
                }
            )
            for box in fboxes:
                box_anns.append(
                    {
                        'id': len(box_anns),
                        'image_id': img_id,
                        'category_id': 1,
                        'bbox': box,
                    }
                )

    return {
        'images': img_anns,
        'annotations': box_anns,
        'categories': [{'id': 1, 'name': 'person'}],
    }


def df2coco(df, video_dir):

    images = []
    annotations = []
    # imgW, imgH = Image.open(video_dir / 'imgs' / '0000.jpg').size
    imgW, imgH = 360, 288

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
    parser.add_argument('--mmpt_train', type=Path, default='../EPFL/dataset/train')
    parser.add_argument('--mmpt_validation', type=Path, default='../EPFL/dataset/validation')
    parser.add_argument('--jsons', type=Path, default='./jsons')
    parser.add_argument('--videos', type=Path, default='./videos/')
    args = parser.parse_args()

    assert args.mmpt_train.exists()
    assert args.mmpt_validation.exists()

    # Prepare jsons
    if not os.path.exists(args.jsons):
        args.jsons.mkdir(parents=True)

    mmpt_train_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    for gt_path in args.mmpt_train.glob('**/**/gt/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        mmpt_train_data['images'].extend(imgs)
        mmpt_train_data['annotations'].extend(anns)
    with open(args.jsons / 'epfl_train.json', 'w') as f:
        json.dump(normalize_ids(mmpt_train_data), f)
    print('mmpt_train: {} {}'.format(len(mmpt_train_data['images']), len(mmpt_train_data['annotations'])))

    mmpt_validation_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    for gt_path in args.mmpt_validation.glob('**/**/gt/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        mmpt_validation_data['images'].extend(imgs)
        mmpt_validation_data['annotations'].extend(anns)
    with open(args.jsons / 'epfl_validation.json', 'w') as f:
        json.dump(normalize_ids(mmpt_validation_data), f)
    print('mmpt_validation: {} {}'.format(len(mmpt_validation_data['images']), len(mmpt_validation_data['annotations'])))
