import os, tarfile

from tqdm  import tqdm
from utils import init_path, check_setting
import shutil
from pathlib import Path

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR = cfg.PATH.INPUT_PATH

def main():
    for scene_path in Path(INPUT_DIR).glob("*"):
        if not os.path.isdir(scene_path): continue
        results_filename = os.path.join(INPUT_DIR, f"track_{scene_path.stem}.txt")
        with open(results_filename, "w+") as f:
            for res_path in scene_path.glob("*/res.txt"):
                with open(res_path, "r") as tmp_f:
                    for line in tmp_f.readlines():
                        f.write(line)
        if os.path.exists(f"/home/ykwei/MCMT-SU/EPFL/eval/track_{scene_path.stem}.txt"):
            os.remove(f"/home/ykwei/MCMT-SU/EPFL/eval/track_{scene_path.stem}.txt")
        shutil.copyfile(results_filename, f"/home/ykwei/MCMT-SU/EPFL/eval/track_{scene_path.stem}.txt")

if __name__ == "__main__":
    main()