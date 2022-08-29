import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from bounding_box import bounding_box as bb
import os
from tqdm  import tqdm
from datetime import datetime


def get_path_scene_cam(camera_name):
    scene_dict = {
        'S01' :{
            'path':'train',
            'cams':('c001','c002','c003','c004','c005')
        },
        'S02' :{
            'path':'validation',
            'cams':('c006','c007','c008','c009')
        },
        'S03' :{
            'path':'train',
            'cams':('c010','c011','c012','c013','c014','c015')
        },
        'S04' :{
            'path':'train',
            'cams':('c016','c017','c018','c019','c020','c021','c022','c023','c024','c025',
                    'c026','c027','c028','c029','c030','c031','c032','c033','c034','c035',
                    'c036','c037','c038','c039','c040')
        },
        'S05':{
            'path':'validation',
            'cams': ('c010','c016','c017','c018','c019','c020','c021','c022','c023',
                    'c024','c025','c026','c027','c028','c029','c033','c034','c035','c036'),
        },
        'S06' :{
            'path':'test',
            'cams': ('c041','c042','c043','c044','c045','c046')
        }
    }
    for scene_id,cam_megs in scene_dict.items():
        cam_ids = cam_megs['cams']
        cam_path = cam_megs['path']
        if camera_name in cam_ids:
            return cam_path, scene_id, camera_name

def generate_video(df, images_path, camera_name, mode, without_su):
    frames_group = df.groupby('fid')
    cap = cv2.VideoCapture(os.path.join(Path(images_path).parent, 'vdo.avi'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
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
    parser.add_argument('--label', type=Path, help='det/sct/det_yolov5 dir of label', default='/home')
    parser.add_argument('--without_su', action='store_true')
    args = parser.parse_args()
    
    assert args.cam in [ 'c%03d' % i for i in range(1, 47) ] + ['train', 'validation', 'test']
    assert args.mode in ['sct', 'det', 'res', 'sct_post', 'det_yolov5']

    cam = args.cam
    mode = args.mode
    label_dir = args.label
    without_su = args.without_su

    
    if cam in ['train', 'validation', 'test']:
        print("輸入資料夾，產生一整個資料的影片")
        if mode == 'sct_post':
            dir_path = Path(os.path.join('mcmt/dataset', cam))
            for label_path  in dir_path.glob('*/*/c04*'):
                images_path = os.path.join(label_path.parent, 'imgs')
                label_path = str(label_path)
                camera_name = label_path.split('/')[-2]
                if not without_su:
                    df = pd.read_csv(label_path, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'ul', 'ut', 'ur', 'ub']
                else:
                    df = pd.read_csv(label_path, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h']
                generate_video(df, images_path, camera_name, mode, without_su)
        elif mode == 'det':
            dir_path = label_dir
            for label_path  in dir_path.glob('*.csv'):
                camera_name = label_path.stem
                cam_path, scene_id, camera_name = get_path_scene_cam(camera_name)
                images_path = os.path.join('mcmt/dataset', cam_path, scene_id, camera_name, 'imgs')
                df = pd.read_csv(label_path)
                generate_video(df, images_path, camera_name, mode, without_su)
        elif mode == 'det_yolov5':
            without_su = True
            dir_path = label_dir
            for label_path  in dir_path.glob('**/**/det/det_yolov5.txt'):
                camera_name = label_path.parent.parent.stem
                cam_path, scene_id, camera_name = get_path_scene_cam(camera_name)
                images_path = os.path.join('mcmt/dataset', cam_path, scene_id, camera_name, 'imgs')
                df = pd.read_csv(label_path)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
                generate_video(df, images_path, camera_name, mode, without_su)
        elif mode == 'sct':
            dir_path = label_dir
            for label_path in dir_path.glob('*.txt'):
                if 'feature' in label_path.stem:
                    continue
                camera_name = label_path.stem
                cam_path, scene_id, camera_name = get_path_scene_cam(camera_name)
                images_path = os.path.join('mcmt/dataset', cam_path, scene_id, camera_name, 'imgs')
                if not without_su:
                    df = pd.read_csv(label_path, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub']
                else:
                    df = pd.read_csv(label_path, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
                generate_video(df, images_path, camera_name, mode, without_su)
        elif mode == 'res':
            without_su = True
            dir_path = Path(os.path.join('mcmt/dataset', cam))
            for label_path  in dir_path.glob('*/*/res.txt'):
                images_path = os.path.join(label_path.parent, 'imgs')
                label_path = str(label_path)
                camera_name = label_path.split('/')[-2]
                df = pd.read_csv(label_path, header=None)
                df.columns = ['cid', 'tag', 'fid', 'x', 'y', 'w', 'h', 'xworld', 'yworld']
                generate_video(df, images_path, camera_name, mode, without_su)

    else:
        cam_path, scene_id, camera_name = get_path_scene_cam(cam)
        images_path = os.path.join('mcmt/dataset', cam_path, scene_id, camera_name, 'imgs')
        if mode == 'det':
            label_path = os.path.join(label_dir, f'{cam}.csv')
            if not Path(label_path).exists():
                print(f'{label_path} DOES NOT EXIST')
                exit()
            df = pd.read_csv(label_path)
        elif mode == 'sct':
            label_path = os.path.join(label_dir, f'{cam}.txt')
            if not Path(label_path).exists():
                print(f'{label_path} DOES NOT EXIST')
                exit()
            if not without_su:
                df = pd.read_csv(label_path, header=None)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', 'ul', 'ut', 'ur', 'ub']
            else:
                df = pd.read_csv(label_path, header=None)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
        elif mode == 'sct_post':
            label_path = os.path.join('mcmt/dataset', cam_path, scene_id, camera_name, f'{cam}_post.txt')
            if not Path(label_path).exists():
                print(f'{label_path} DOES NOT EXIST')
                exit()
            if not without_su:
                df = pd.read_csv(label_path, header=None)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 'ul', 'ut', 'ur', 'ub']
            else:
                df = pd.read_csv(label_path, header=None)
                df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h']
        elif mode == 'res':
            label_path = os.path.join('mcmt/dataset', cam_path, scene_id, camera_name, 'res.txt')
            if not Path(label_path).exists():
                print(f'{label_path} DOES NOT EXIST')
                exit()
            if not without_su:
                print("res.txt還沒寫有su版本")
                exit()
            else:
                df = pd.read_csv(label_path, header=None)
                df.columns = ['cid', 'tag', 'fid', 'x', 'y', 'w', 'h', 'xworld', 'yworld']

        generate_video(df, images_path, camera_name, mode, without_su)
