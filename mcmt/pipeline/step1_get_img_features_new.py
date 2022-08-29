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

ts_dict = get_timestamp_dict(os.path.join(ROOT_DIR, "cam_timestamp"))
fps_dict = get_fps_dict(INPUT_DIR)

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

def analysis_transfrom_mat(cali_path):
    first_line = open(cali_path).readlines()[0].strip('\r\n')
    cols = first_line.lstrip('Homography matrix: ').split(';')
    transfrom_mat = np.ones((3, 3))
    for i in range(3):
        values_string = cols[i].split()
        for j in range(3):
            value = float(values_string[j])
            transfrom_mat[i][j] = value
    inv_transfrom_mat = np.linalg.inv(transfrom_mat)
    return inv_transfrom_mat

def prepare_data():
    data_dict = dict()
    
    for scene_dir in tqdm(os.listdir(INPUT_DIR), desc="Loading Data"):
        if not scene_dir.startswith("S0"):
            continue
        for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
            if not camera_dir.startswith("c0"):
                continue
       
            feature_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, f"{cfg.SCT}_{cfg.DETECTION}_all_features.txt")
            if os.path.exists(feature_file):
                os.remove(feature_file)

            cali_path = os.path.join(INPUT_DIR, scene_dir, camera_dir, 'calibration.txt')
            trans_mat = analysis_transfrom_mat(cali_path)
            sct_file = os.path.join(SCT_DIR, f"{camera_dir}.txt")
            sct_results = read_sct_file(sct_file)
            sct_features_file = os.path.join(SCT_DIR, f"{camera_dir}_features.txt")
            sct_features = read_sct_features_file(sct_features_file)
            data_dict[scene_dir + '/' + camera_dir] = [sct_results, sct_features, trans_mat]
            
    return data_dict

def get_all_feature_file(scene_camera, sct_results, sct_features, trans_mat):
    for fid_id_key in tqdm(sct_results, desc=f"Processing Features For {scene_camera}:"):
        scene_dir, camera_dir = scene_camera.split("/")
        fps = fps_dict[camera_dir]
        start_ts = ts_dict[camera_dir]
        path = os.path.join(INPUT_DIR, scene_dir, camera_dir)
        sct_result = sct_results[fid_id_key]
        frame_id, id = fid_id_key.split("_")
        ts = (1. / fps) * int(frame_id) + start_ts
        if WITH_SU:
            box = sct_result.split(",")[0:4]
        else:
            box = sct_result.split(",")
        coor = [int(float(box[0])) + int(float(box[2]))/2, int(float(box[1])) + int(float(box[3]))/2, 1]
        GPS_coor = np.dot(trans_mat, coor)
        GPS_coor = GPS_coor / GPS_coor[2]
        reid_feature = sct_features[fid_id_key]
        '''
        WITH_SU
        True: fid, id, left, top, width, height, ul, ut, ur, ub, ts, GPS0, GPS1, feature(2048)
        False: fid, id, left, top, width, height, ts, GPS0, GPS1, feature(2048)
        '''
        with open(os.path.join(path, f'{cfg.SCT}_{cfg.DETECTION}_all_features.txt'), 'a+') as f:
            line = frame_id + "," + id + "," + sct_result + "," + str(ts) + "," + str(GPS_coor[0]) + "," + str(GPS_coor[1]) \
                    + "," + reid_feature + "\n"
            f.write(line)

if __name__ == "__main__":
    data_dict    = prepare_data()

    for key in data_dict:
        sct_results, sct_features, trans_mat = data_dict[key]
        get_all_feature_file(key, sct_results, sct_features, trans_mat)