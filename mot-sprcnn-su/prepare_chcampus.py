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
    imgW, imgH = 1920, 1080

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
    parser.add_argument('--ch', type=Path, default='../mmpt/CrowdHuman/')
    parser.add_argument('--campus_train', type=Path, default='../campus/dataset/train')
    parser.add_argument('--campus_validation', type=Path, default='../campus/dataset/validation')
    parser.add_argument('--jsons', type=Path, default='./jsons')
    parser.add_argument('--videos', type=Path, default='./videos/')
    args = parser.parse_args()

    assert args.ch.exists()
    assert args.campus_train.exists()
    assert args.campus_validation.exists()

    # Prepare jsons
    if not os.path.exists(args.jsons):
        args.jsons.mkdir(parents=True)

    ch_data = load_odgt(args.ch / 'annotation_train.odgt')
    with open(args.jsons / 'ch.json', 'w') as f:
        json.dump(normalize_ids(ch_data), f)
    print('ch: {} {}'.format(len(ch_data['images']), len(ch_data['annotations'])))

    campus_train_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    for gt_path in args.campus_train.glob('**/**/gt/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        campus_train_data['images'].extend(imgs)
        campus_train_data['annotations'].extend(anns)
    with open(args.jsons / 'campus_train.json', 'w') as f:
        json.dump(normalize_ids(campus_train_data), f)
    print('campus_train: {} {}'.format(len(campus_train_data['images']), len(campus_train_data['annotations'])))

    campus_validation_data = {
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'vehicle'}],
    }
    for gt_path in args.campus_validation.glob('**/**/gt/gt.txt'):
        df = pd.read_csv(gt_path, header=None)
        df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
        imgs, anns = df2coco(df, gt_path.parent.parent)
        campus_validation_data['images'].extend(imgs)
        campus_validation_data['annotations'].extend(anns)
    with open(args.jsons / 'campus_validation.json', 'w') as f:
        json.dump(normalize_ids(campus_validation_data), f)
    print('campus_validation: {} {}'.format(len(campus_validation_data['images']), len(campus_validation_data['annotations'])))
    
    chcampus = {
        'images': [*ch_data['images'], *campus_train_data['images']],
        'annotations': [*ch_data['annotations'], *campus_train_data['annotations']],
        'categories': [{'id': 1, 'name': 'person'}],
    }
    with open(args.jsons / 'chcampus.json', 'w') as f:
        json.dump(normalize_ids(chcampus), f)
    print('chcampus: {} {}'.format(len(chcampus['images']), len(chcampus['annotations'])))