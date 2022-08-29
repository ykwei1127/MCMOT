import os, tarfile

from tqdm  import tqdm
from utils import init_path, check_setting
import shutil

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR = cfg.PATH.INPUT_PATH


def main():
    camera_dirs = list()
    for scene_dir in os.listdir(INPUT_DIR):
        if scene_dir.startswith("S0"):
            # if scene_dir != "S02":
            #     continue
            if scene_dir != "S05":
                continue
            for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
                if camera_dir.startswith("c0"):
                    camera_dirs.append(os.path.join(INPUT_DIR, scene_dir, camera_dir))
    
    results_filename = os.path.join(INPUT_DIR, "track1.txt")
    with open(results_filename, "w+") as f:
        for camera_dir in camera_dirs:
            tmp_result_file = os.path.join(camera_dir, "res.txt")
            with open(tmp_result_file, "r") as tmp_f:
                for line in tmp_f.readlines():
                    f.write(line)
    
    if INPUT_DIR == "/home/ykwei/MCMT-SU/mcmt/dataset/validation":
        if os.path.exists("/home/ykwei/MCMT-SU/mcmt/eval/track1.txt"):
            os.remove("/home/ykwei/MCMT-SU/mcmt/eval/track1.txt")
        shutil.copyfile(results_filename, "/home/ykwei/MCMT-SU/mcmt/eval/track1.txt")

    # tar_filename = os.path.join(INPUT_DIR, "track1.tar.gz")
    # tar = tarfile.open(tar_filename, "w:gz")
    # tar.add(results_filename)
    # tar.close()

if __name__ == "__main__":
    main()