import copy
import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--EPFL_train', type=Path, default='../EPFL/dataset/train')
    # parser.add_argument('--EPFL_validation', type=Path, default='../EPFL/dataset/validation')
    parser.add_argument('--EPFL_test', type=Path, default='../EPFL/dataset2/test')
    # parser.add_argument('--EPFL_temp', type=Path, default='../EPFL/dataset/temp')
    parser.add_argument('--videos', type=Path, default='./videos/')
    args = parser.parse_args()

    # assert args.EPFL_train.exists()
    # assert args.EPFL_validation.exists()
    assert args.EPFL_test.exists()
    # assert args.EPFL_temp.exists()

    # Prepare videos
    # args.videos.mkdir(parents=True)
    # (args.videos / 'EPFL_train').mkdir()
    # (args.videos / 'EPFL_validation').mkdir()
    (args.videos / 'EPFL_test').mkdir()
    # (args.videos / 'EPFL_temp').mkdir()

    # for video_dir in args.mmpt_train.glob('*/*'):
    #     video_name = video_dir.stem
    #     paths = sorted(list(video_dir.glob('imgs/*.jpg')))
    #     fids = [int(p.stem) for p in paths]
    #     df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
    #     df_imgs.to_csv(args.videos / 'mmpt_train' / f'{video_name}.imgs', index=None)
    #     if (video_dir / 'gt' / 'gt.txt').exists():
    #         df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
    #         df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
    #         df_true.to_csv(args.videos / 'mmpt_train' / f'{video_name}.gt', index=None)
            
    # for video_dir in args.mmpt_validation.glob('*/*'):
    #     video_name = video_dir.stem
    #     paths = sorted(list(video_dir.glob('imgs/*.jpg')))
    #     fids = [int(p.stem) for p in paths]
    #     df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
    #     df_imgs.to_csv(args.videos / 'mmpt_validation' / f'{video_name}.imgs', index=None)
    #     if (video_dir / 'gt' / 'gt.txt').exists():
    #         df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
    #         df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
    #         df_true.to_csv(args.videos / 'mmpt_validation' / f'{video_name}.gt', index=None)

    for video_dir in args.EPFL_test.glob('*/*'):
        video_name = video_dir.stem
        paths = sorted(list(video_dir.glob('imgs/*.jpg')))
        fids = [int(p.stem) for p in paths]
        df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
        df_imgs.to_csv(args.videos / 'EPFL_test' / f'{video_name}.imgs', index=None)
        if (video_dir / 'gt' / 'gt.txt').exists():
            df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
            df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
            df_true.to_csv(args.videos / 'EPFL_test' / f'{video_name}.gt', index=None)

    # for video_dir in args.campus_temp.glob('*/*'):
    #     video_name = video_dir.stem
    #     paths = sorted(list(video_dir.glob('imgs/*.jpg')))
    #     fids = [int(p.stem) for p in paths]
    #     df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
    #     df_imgs.to_csv(args.videos / 'campus_temp' / f'{video_name}.imgs', index=None)
    #     if (video_dir / 'gt' / 'gt.txt').exists():
    #         df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
    #         df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
    #         df_true.to_csv(args.videos / 'campus_temp' / f'{video_name}.gt', index=None)