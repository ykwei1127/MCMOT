import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os
from tqdm  import tqdm
import json
import shutil


def normalize_cameraid():
    base_path = Path("dataset/train")
    base_path2 = Path("dataset/validation")
    base_path3 = Path("dataset/test")
    i = 1
    for scene_path in sorted(base_path.glob("*")):
        scene_id = scene_path.stem
        for camera_path in sorted(scene_path.glob("*")):
            new_cameraid_path = os.path.join(camera_path.parent, f"c{str(i).zfill(3)}")
            os.rename(camera_path, new_cameraid_path)
            i += 1
    for scene_path in sorted(base_path2.glob("*")):
        scene_id = scene_path.stem
        for camera_path in sorted(scene_path.glob("*")):
            new_cameraid_path = os.path.join(camera_path.parent, f"c{str(i).zfill(3)}")
            os.rename(camera_path, new_cameraid_path)
            i += 1
    for scene_path in sorted(base_path3.glob("*")):
        scene_id = scene_path.stem
        for camera_path in sorted(scene_path.glob("*")):
            new_cameraid_path = os.path.join(camera_path.parent, f"c{str(i).zfill(3)}")
            os.rename(camera_path, new_cameraid_path)
            i += 1

normalize_cameraid()