import torch, os, re, time
import torch.multiprocessing as mp
import numpy as np

from PIL              import Image
from tqdm             import tqdm 
from utils.utils      import get_fps_dict, get_timestamp_dict
from utils            import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR   = cfg.PATH.INPUT_PATH
ROOT_DIR    = cfg.PATH.ROOT_PATH
DEVICE      = cfg.DEVICE.TYPE
GPUS        = cfg.DEVICE.GPUS
BATCH_SIZE  = cfg.REID.BATCH_SIZE
NUM_WORKERS = 4 #1
SCT_DIR     = cfg.PATH.SCT_PATH
WITH_SU     = cfg.WITH_SU

def read_sct_file(file):
    f = open(file, "r")
    results = dict()
    for line in f.readlines():
        l = line.strip("\n").split(",")
        frame_id = l[0]
        id = l[1]
        if WITH_SU:
            box = ",".join([l[index] for index in [2,3,4,5,7,8,9,10]]) # bounding box + spatial uncertainty
        else:
            box = ",".join(l[2:6])
        results[frame_id + '_' + id] = box
    
    return results

def read_sct_features_file(file):
    f = open(file, "r")
    results = dict()
    for line in f.readlines():
        l = line.strip("\n").split(",")
        frame_id = l[0]
        id = l[1]
        feature = ",".join(l[2:])
        results[frame_id + '_' + id] = feature
    
    return results

def prepare_data():
    data_dict = dict()
    
    for scene_dir in tqdm(os.listdir(INPUT_DIR), desc="Loading Data"):
        if scene_dir not in ["Auditorium", "Garden1", "Garden2", "Parkinglot"]:
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
       
            feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"{cfg.SCT}_{cfg.DETECTION}_all_features.txt")
            if os.path.exists(feature_file):
                os.remove(feature_file)

            sct_file = os.path.join(SCT_DIR, f"{camera_dir}.txt")
            sct_results = read_sct_file(sct_file)
            sct_features_file = os.path.join(SCT_DIR, f"{camera_dir}_features.txt")
            sct_features = read_sct_features_file(sct_features_file)
            data_dict[scene_dir + '/' + camera_dir] = [sct_results, sct_features]
            
    return data_dict

def get_all_feature_file(scene_camera, sct_results, sct_features):
    for fid_id_key in tqdm(sct_results, desc=f"Processing Features For {scene_camera}:"):
        scene_dir, camera_dir = scene_camera.split("/")
        path = os.path.join(INPUT_DIR, scene_dir, camera_dir)
        sct_result = sct_results[fid_id_key]
        frame_id, id = fid_id_key.split("_")
        reid_feature = sct_features[fid_id_key]
        '''
        WITH_SU
        True: fid, id, left, top, width, height, ul, ut, ur, ub, ts, GPS0, GPS1, feature(2048)
        False: fid, id, left, top, width, height, ts, GPS0, GPS1, feature(2048)

        New: fid, id, left, top, width, height, ul, ut, ur, ub, feature(2048)
        or:  fid, id, left, top, width, height, feature(2048)
        '''
        with open(os.path.join(path, f'{cfg.SCT}_{cfg.DETECTION}_all_features.txt'), 'a+') as f:
            line = frame_id + "," + id + "," + sct_result + "," + reid_feature + "\n"
            f.write(line)

if __name__ == "__main__":
    data_dict    = prepare_data()

    for key in data_dict:
        sct_results, sct_features = data_dict[key]
        get_all_feature_file(key, sct_results, sct_features)