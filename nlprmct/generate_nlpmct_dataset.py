import argparse
from operator import index
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os
from tqdm  import tqdm
import json
import shutil
from multiprocessing import Pool


def video2images(video_file, save_path):
    vc = cv2.VideoCapture(video_file)
    fps = vc.get(cv2.CAP_PROP_FPS)
    num_frame = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    while True:
        rval, frame = vc.read()
        if not rval:
            break
        name = "{:05d}.jpg".format(frame_count)
        cv2.imwrite(os.path.join(save_path,name),frame)
        frame_count+=1
    vc.release()

def video2images_api(args):
    video_file = args[0]
    save_path = args[1]
    video2images(video_file,save_path)

def move_videos():
    base_path = Path("dataset/video")
    dataset_path = Path("dataset/temp")
    args_list = []
    for scene in base_path.glob("*"):
        scene_path = os.path.join(dataset_path, scene.stem)
        print(scene_path)
        if not os.path.isdir(scene_path):
            os.mkdir(scene_path)
        for video_path in tqdm(scene.glob("Cam*.avi"), desc=scene.stem):
            camera_id = video_path.stem
            camera_path = os.path.join(scene_path, camera_id)
            if not os.path.isdir(camera_path):
                os.mkdir(camera_path)
            imgs_path = os.path.join(camera_path, "imgs")
            if not os.path.isdir(imgs_path):
                os.mkdir(imgs_path)
            gt_path = os.path.join(camera_path, "gt")
            if not os.path.isdir(gt_path):
                os.mkdir(gt_path)
            args_list.append([str(video_path), imgs_path])
    n_jobs = 12
    pool = Pool(n_jobs)
    pool.map(video2images_api, args_list)
    pool.close()

def get_labels():
    base_path = Path("dataset/temp")
    label_dir = Path("dataset/annotation")
    for scene_path in base_path.glob("*"):
        scene_id = scene_path.stem
        max_id = -1
        for camera_path in scene_path.glob("*"):
            camera_id = camera_path.stem
            label_file = os.path.join(label_dir, scene_id, f"{camera_id}.dat")
            df = pd.read_csv(label_file, sep="\s+", header=None)
            df.columns = ['cid', 'fid', 'tag', 'x', 'y', 'w', 'h']
            column = df["tag"]
            if column.max() > max_id:
                max_id = column.max()
        print(scene_id, max_id)

    bias = [0, 235, 490, 504]
    for i,scene_path in enumerate(base_path.glob("*")):
        print(i)
        scene_id = scene_path.stem
        for camera_path in scene_path.glob("*"):
            camera_id = camera_path.stem
            gt_path = os.path.join(camera_path, "gt", "gt.txt")
            label_file = os.path.join(label_dir, scene_id, f"{camera_id}.dat")
            with open(gt_path, "w") as f:
                df = pd.read_csv(label_file, sep="\s+", header=None)
                df.columns = ['cid', 'fid', 'tag', 'x', 'y', 'w', 'h']
                for index, row in df.iterrows():
                    # [frame, ID, left, top, width, height, 1, -1, -1, -1]
                    frame_id = row["fid"]
                    id = row["tag"] + bias[i]
                    left = row["x"]
                    top = row["y"]
                    width = row["w"]
                    height = row["h"]
                    line = str(frame_id)+","+str(id)+","+str(left)+","+str(top)+","+str(width)+","+str(height)+ \
                            "," + "1" + "," + "-1" + "," + "-1" + "," + "-1" + "\n"
                    f.write(line)

def split_data():
    base_path = Path("dataset/temp")
    for scene in base_path.glob("*"):
        min_len = 10000000
        for imgs_dir in scene.glob("*/imgs"):
            if len(os.listdir(imgs_dir)) < min_len:
                min_len = len(os.listdir(imgs_dir))

        train_len = int(min_len * 0.15)
        valid_len = int(min_len * 0.2) - train_len
        print(train_len, valid_len)
        for split in ["train", "validation", "test"]:
            split_scene_path = os.path.join("dataset", split, scene.stem)
            print(split_scene_path)
            if not os.path.isdir(split_scene_path):
                os.mkdir(split_scene_path)
            for camera_dir in scene.glob("*"):
                print(camera_dir)
                camera_id = camera_dir.stem
                camera_path = os.path.join(split_scene_path, camera_id)
                if not os.path.isdir(camera_path):
                    os.mkdir(camera_path)
                imgs_path = os.path.join(camera_path, "imgs")
                if not os.path.isdir(imgs_path):
                    os.mkdir(imgs_path)
                gt_path = os.path.join(camera_path, "gt")
                if not os.path.isdir(gt_path):
                    os.mkdir(gt_path)
                if split == "train":
                    old_gt_file = os.path.join(camera_dir, "gt", "gt.txt")
                    df = pd.read_csv(old_gt_file, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
                    new_gt_file = os.path.join(gt_path, "gt.txt")
                    for i in range(0, train_len):
                        img_name = f"{str(i).zfill(5)}.jpg"
                        source = os.path.join(camera_dir, "imgs", img_name)
                        destination = os.path.join(imgs_path, img_name)
                        shutil.copy(source, destination)
                        if i not in df.groupby('fid').groups.keys():
                            continue
                        frames_group = df.groupby('fid').get_group(i)
                        frames_group.to_csv(new_gt_file, mode='a', index=False, header=False)
                elif split == "validation":
                    old_gt_file = os.path.join(camera_dir, "gt", "gt.txt")
                    df = pd.read_csv(old_gt_file, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
                    new_gt_file = os.path.join(gt_path, "gt.txt")
                    for i in range(train_len, train_len+valid_len):
                        img_name = f"{str(i).zfill(5)}.jpg"
                        source = os.path.join(camera_dir, "imgs", img_name)
                        destination = os.path.join(imgs_path, img_name)
                        shutil.copy(source, destination)
                        if i not in df.groupby('fid').groups.keys():
                            continue
                        frames_group = df.groupby('fid').get_group(i)
                        frames_group.to_csv(new_gt_file, mode='a', index=False, header=False)
                elif split == "test":
                    ori_img_dir = os.path.join(camera_dir, "imgs")
                    all_len = len(os.listdir(ori_img_dir))
                    ori_img_dir = os.path.join(camera_dir, "imgs")
                    old_gt_file = os.path.join(camera_dir, "gt", "gt.txt")
                    df = pd.read_csv(old_gt_file, header=None)
                    df.columns = ['fid', 'tag', 'x', 'y', 'w', 'h', 's', '1', '2', '3']
                    new_gt_file = os.path.join(gt_path, "gt.txt")
                    for i in range(train_len+valid_len, all_len):
                        img_name = f"{str(i).zfill(5)}.jpg"
                        source = os.path.join(camera_dir, "imgs", img_name)
                        destination = os.path.join(imgs_path, img_name)
                        shutil.copy(source, destination)
                        if i not in df.groupby('fid').groups.keys():
                            continue
                        frames_group = df.groupby('fid').get_group(i)
                        frames_group.to_csv(new_gt_file, mode='a', index=False, header=False)      

# move_videos()
# get_labels()
split_data()