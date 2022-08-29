import copy
import json
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlprmct_test', type=Path, default='../nlprmct/dataset/test')
    parser.add_argument('--videos', type=Path, default='./videos/')
    args = parser.parse_args()


    assert args.nlprmct_test.exists()

    # Prepare videos
    (args.videos / 'nlprmct_test').mkdir()

    for video_dir in args.nlprmct_test.glob('*/*'):
        video_name = video_dir.stem
        paths = sorted(list(video_dir.glob('imgs/*.jpg')))
        fids = [int(p.stem) for p in paths]
        df_imgs = pd.DataFrame({'fid': fids, 'path': paths})
        df_imgs.to_csv(args.videos / 'nlprmct_test' / f'{video_name}.imgs', index=None)
        if (video_dir / 'gt' / 'gt.txt').exists():
            df_true = pd.read_csv(video_dir / 'gt' / 'gt.txt', header=None)
            df_true.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'conf', 'xx', 'yy', 'zz']
            df_true.to_csv(args.videos / 'nlprmct_test' / f'{video_name}.gt', index=None)