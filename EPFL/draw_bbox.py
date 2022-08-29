import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from bounding_box import bounding_box as bb
import os
from tqdm  import tqdm
from datetime import datetime

def generate_video(df, images_path, camera_name, mode, without_su):
    frames_group = df.groupby('fid')
    fps = 25
    width = 360
    height = 288
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    if not os.path.isdir("videos"):
        os.mkdir("videos")
    video = cv2.VideoWriter(f'videos/{camera_name}_{mode}_{timestamp}.avi', fourcc, fps, (width, height))

    files = os.listdir(images_path)
    files.sort(key=lambda x:int(x[:-4]))
    for img_file in tqdm(files, desc=f'{camera_name}_{mode}'):
        img_path = os.path.join(images_path, img_file)
        img_name = img_file[:-4]
        if int(img_name) not in frames_group.groups.keys():
            img = cv2.imread(img_path)
            video.write(img)
            continue
        img = cv2.imread(img_path)
        frame_df = frames_group.get_group(int(img_name))
        for index, row in frame_df.iterrows():
            if row["x"] < 0: row["x"] = 0
            if row["y"] < 0: row["y"] = 0
            if without_su:
                bb.add(img, row['x'], row['y'], row['x']+row['w'], row['y']+row['h'], str(int(row['tag'])))
            else:
                color = bb.add(img, row['x'], row['y'], row['x']+row['w'], row['y']+row['h'], str(int(row['tag'])))
                cx = row['x']+row['w']/2
                cy = row['y']+row['h']/2
                start_point = (int(cx-row['ul']), int(cy))
                end_point = (int(cx+row['ur']), int(cy))
                img = cv2.line(img, start_point, end_point, color, 2)
                start_point = (int(cx), int(cy-row['ut']))
                end_point = (int(cx), int(cy+row['ub']))
                img = cv2.line(img, start_point, end_point, color, 2)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--label', type=Path, help='gt/sct/det dir of label', default='/home')
    parser.add_argument('--without_su', action='store_true')
    args = parser.parse_args()
    
    assert args.cam in ['train', 'validation', 'test', 'temp']
    assert args.mode in ['gt', 'sct', 'det', 'res']

    cam = args.cam
    mode = args.mode
    label_dir = args.label
    without_su = args.without_su

    
    if cam in ['train', 'validation', 'test', 'temp']:
        print("輸入資料夾，產生一整個資料的影片")
        if mode == 'gt':
            without_su = True
            dir_path = label_dir
            for label_path  in dir_path.glob('**/gt/gt.txt'):
                camera_name = label_path.parent.parent.stem
                images_path = os.path.join(label_path.parent.parent, 'imgs')
                df = pd.read_csv(label_path)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
                generate_video(df, images_path, camera_name, mode, without_su)
        elif mode == "sct":
            dir_path = label_dir
            for label_path  in dir_path.glob('*'):
                if "features" in label_path.stem: continue
                camera_id = label_path.stem
                for a in Path("dataset2/test").glob(f"*/{camera_id}"):
                    images_path = os.path.join(a, "imgs")
                df = pd.read_csv(label_path)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub']
                generate_video(df, images_path, camera_id, mode, without_su)
        elif mode == "det":
            dir_path = label_dir
            for label_path  in dir_path.glob('*'):
                camera_id = label_path.stem
                for a in Path("dataset/test").glob(f"*/{camera_id}"):
                    images_path = os.path.join(a, "imgs")
                df = pd.read_csv(label_path)
                generate_video(df, images_path, camera_id, mode, without_su)
        elif mode == 'res':
            without_su = True
            dir_path = Path(os.path.join('dataset', cam))
            for label_path  in dir_path.glob('*/*/res.txt'):
                images_path = os.path.join(label_path.parent, 'imgs')
                label_path = str(label_path)
                camera_id = label_path.split('/')[-2]
                df = pd.read_csv(label_path)
                df.columns = ['cid', 'tag', 'fid', 'x', 'y', 'w', 'h', '1', '2']
                generate_video(df, images_path, camera_id, mode, without_su)
                
                