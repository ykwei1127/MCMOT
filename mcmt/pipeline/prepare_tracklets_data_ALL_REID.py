import torch, os, re, time
import torch.multiprocessing as mp
import numpy as np

from torch.utils.data import DataLoader, Dataset
from PIL              import Image
from tqdm             import tqdm 
from utils.reid       import build_transform, build_model
from utils            import init_path, check_setting

init_path()

from config import cfg
from reid.config import cfg as cfg2
from reid.reid_inference.reid_model import build_reid_model, build_reid_model_2, build_reid_model_3
from utils.reid       import build_transform_2
from sklearn import preprocessing
from torchvision.utils import save_image

check_setting(cfg)

# INPUT_DIR   = os.path.join(cfg.PATH.ROOT_PATH, "train")
INPUT_DIR   = os.path.join(cfg.PATH.ROOT_PATH, "validation")
DEVICE      = cfg.DEVICE.TYPE
GPUS        = cfg.DEVICE.GPUS
BATCH_SIZE  = cfg.REID.BATCH_SIZE
NUM_WORKERS = 8 # 4

FLIP_FEATURE = cfg.FLIP_FEATURE

class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def read_image(self, img_path):
        got_img = False
        if not os.path.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'.".format(img_path))
                exit()
        return img

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = self.read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, img_path

def collate_fn(batch):
    imgs, path = zip(*batch)
    return torch.stack(imgs, dim=0), path

def prepare_data():
    image_list = list()
    image_path = os.path.join(INPUT_DIR, "gt_images")
    for img_name in os.listdir(image_path):
        image_list.append(os.path.join(image_path, img_name))
            
    return image_list

def progress(task_num, desc, finish_queue):
    now = 0
    pbar = tqdm(total=task_num, desc=desc)
    while True:
        if now == task_num:
            break
        if finish_queue.empty():
            continue
        finish_queue.get()
        pbar.update()
        now += 1

def main(device, data_queue, stop, write_lock, finish_queue):
    model, reid_cfg = build_reid_model(cfg2)
    model.to(device)
    model = model.eval()

    model_2, reid_cfg = build_reid_model_2(cfg2)
    model_2.to(device)
    model_2 = model_2.eval()

    model_3, reid_cfg = build_reid_model_3(cfg2)
    model_3.to(device)
    model_3 = model_3.eval()

    finish_queue.get()
    while not data_queue.empty() or not stop.value:
        if data_queue.empty():
            continue
        data, paths = data_queue.get()

        with torch.no_grad():
            data = data.to(device)
            if FLIP_FEATURE:
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(data.size(3) - 1, -1, -1).long().to(device)
                        data = data.index_select(3, inv_idx)
                        feat1 = model(data)
                        feat1_2 = model_2(data)
                        feat1_3 = model_3(data)
                    else:
                        feat2 = model(data)
                        feat2_2 = model_2(data)
                        feat2_3 = model_3(data)
                feat = feat2 + feat1
                feat_2 = feat2_2 + feat1_2
                feat_3 = feat2_3 + feat1_3
            else:
                feat = model(data)
                feat_2 = model_2(data)
                feat_3 = model_3(data)

            for i,p in enumerate(paths):
                patch_feature_list = [feat[i].cpu().numpy(), feat_2[i].cpu().numpy(), feat_3[i].cpu().numpy()]
                patch_feature_array = np.array(patch_feature_list)
                patch_feature_array = preprocessing.normalize(patch_feature_array, norm='l2', axis=1)
                patch_feature_mean = np.mean(patch_feature_array, axis=0)

                reid_feat = list(patch_feature_mean)
                # reid_feat = list(feat[i].cpu().numpy())
                reid_feat_str = str(reid_feat)[1:-1].replace(" ", "")
                img_name = p.split('/')[-1][:-4]
                info = img_name.split('_')
                det_id = info[0]
                camera_id = info[1]
                frame_id = info[2]

                write_lock.acquire()
                with open(os.path.join(INPUT_DIR, f'gt_features.txt'), 'a+') as f:
                    line = camera_id + ',' + frame_id + "," + det_id + "," + reid_feat_str + "\n"
                    f.write(line)
                write_lock.release()
        
        finish_queue.put(True)

if __name__ == "__main__":
    devices = list()
    
    if DEVICE == "cuda":
        for gpu in GPUS:
            devices.append(torch.device(DEVICE + ':' + str(gpu)))

    elif DEVICE == "cpu":
        for i in range(4):
            devices.append(torch.device(DEVICE))

    mp.set_start_method("spawn")

    image_list   = prepare_data()
    transforms   = build_transform_2(cfg)
    data_queue   = mp.Queue()
    finish_queue  = mp.Queue()
    stop         = mp.Value("i", False)
    write_lock   = mp.Lock()

    print (f"Create {len(devices)} processes.")

    processes = list()
    for device in devices:
        finish_queue.put(True)
        p = mp.Process(target=main, args=(device, data_queue, stop, write_lock, finish_queue))
        p.start()
        processes.append(p)

    print ("Loading Model...")
    while not finish_queue.empty():
        time.sleep(0.5)
    
    dataset = ImageDataset(image_list, transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn)
    pbar_process = mp.Process(target=progress, args=(len(dataloader), f"Extracting Features", finish_queue))
    pbar_process.start()
    for data, paths in dataloader:
        while not data_queue.empty():
            pass
        data_queue.put([data, paths])
    pbar_process.join()

    stop.value = True

    for p in processes:
        p.join()

    data_queue.close()
    data_queue.join_thread()