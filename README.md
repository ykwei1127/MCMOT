# MCMOT with Spatial Uncertainty
Deep Tracklet Feature Representation via Spatial Uncertainty for Multi-Camera Multi-Object Tracking System

# Get Started
## Installation (Linux)
* python==3.8
* pytorch==1.8
* pytorch-lightning==1.4.9
* [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
* scipy==1.8.1
* opencv-python==4.6.0.66
* imgaug==0.4.0
* pandas==1.1.5
* motmetrics==1.2.5 
* filterpy==1.4.5
* cmake==3.22.5 
* lapsolver==1.1.0
* [torchreid](https://github.com/KaiyangZhou/deep-person-reid)
* others (install whatever is missing)
* `conda env create -f environment.yaml`

## Download Datasets
Download the following datasets and put into the corresponding folders.
```
MCMOT/mcmt/dataset/
MCMOT/PETS09-S2L1/dataset/temp
MCMOT/nlprmct/dataset/temp
MCMOT/EPFL/dataset/temp
MCMOT/campus/dataset/temp
```
* [CityFlowV2 Track1](https://www.aicitychallenge.org/2022-data-and-evaluation/)
* [PETS09-S2L1](http://cs.binghamton.edu/~mrldata/public/PETS2009/S1_L2.tar.bz2)
* [NLPR-MCT](http://www.mct2014.com/Datasets.html)
* [EPFL](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)
* [CAMPUS](https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/)

**Note that the annotaions of PETS09-S2L1, EPFL and CAMPUS are from [here](https://bitbucket.org/merayxu/multiview-object-tracking-dataset/src/master/)**.

After downloading the datasets, run the following commands to generate the data that we can use in the code. (Only for pedestrian datasets.)
```
python generate_*_dataset.py
python normalize_cameraid.py
```

## Petrained Weights
* It should be available at [here](https://github.com/ykwei1127/MCMOT/releases/tag/Weights).

* To train the object detector:
    1. ```p300_coco.ckp```: 300 proposal bounding boxes, trained on COCO
    2. ```p300_ch17.ckpt```: 300 proposal bounding boxes, trained on CrowdHuman + MOT17
    Place them under ```MCMOT/mot-sprcnn-su/weights/```

* For pedestrian ReID model:
    1. ```osnet_x1_0_MS_D_C.pth```
    Place it under ```MCMOT/mot-sprcnn-su/weights/```

* For vehicle ReID model:
    1. ```resnet101_ibn_a_2.pth```
    2. ```resnet101_ibn_a_3.pth```
    3. ```resnext101_ibn_a_2.pth```
    Place them under ```MCMOT/mot-sprcnn-su/reid/reid_model/```

* Trained weights for object detection:
    1. ```version8_epoch=39-step=99639.ckpt```: for CityFlowV2 dataset
    2. ```p300_ch17.ckpt```: for PETS09-S2L1 and EPFL
    3. ```v17e39_pets09.ckpt```: for PETS09-S2L1
    4. ```v20e3_nlprmct.ckpt```: for NLPR-MCT
    5. ```v22e23_epfl2.ckpt```: for EPFL
    6. ```v21e11_campus.ckpt```: for CAMPUS
    Place them under ```MCMOT/mot-sprcnn-su/weights/```

# Single-Camera Tracking
**CityFlowV2 and PETS09-S2L1 are used as examples for the following steps.**

```
cd mot-sprcnn-su/
```

## Prepare Data
**CityFlowV2:**
```
python prepare_coco_train+valid+detrac.py
```

**PETS09-S2L1:**
```
python prepare_ch+pets09.py
```

**If you just want to inference detection, just run the following command:**
```
python prepare_*_videos.py
```
\* can be pets09, nlprmct, etc.


## Object Detector Training (Optional)
**CityFlowV2:**
```
python train.py --train_json jsons/vehicle_train.json --valid_json jsons/vehicle_validation.json --num_proposals 300
```

**PETS09-S2L1:**
```
python train.py --train_json jsons/pets09_train.json --valid_json jsons/pets09_validation.json --num_proposals 300
```


## Object Detector Inference
**Object detection and ReID features extraction.**

**CityFlowV2:**
```
python infer_dets.py --videos videos/aic22_valid --ckpt weights/version8_epoch=39-step=99639.ckpt --dets dets/aic22_valid --score_thresh 0.4 --nms_thresh 0.7 --num_proposals 300
python infer_reid_all.py --videos videos/aic22_valid --dets dets/aic22_valid
```

**PETS09-S2L1:**
```
python infer_dets.py --videos videos/pets09 --ckpt weights/p300_ch17.ckp --dets dets/pets09 --score_thresh 0.4 --nms_thresh 0.7  --num_proposals 300
python infer_osnet.py --videos videos/pets09 --dets dets/pets09
```


## Tracking
**CityFlowV2:**
```
python track_sct+feature.py --videos videos/aic22_valid --dets dets/aic22_valid --outs outs/aic22_valid
```

**PETS09-S2L1:**
```
python track_sct+feature.py --videos videos/pets09 --dets dets/pets09 --outs outs/pets09
```


# Multi-Camera Tracking
## Step 1: get bounding box features of the tracklets
**CityFlowV2:**
```
cd mcmt/
```

**PETS09-S2L1:**
```
cd PETS09-S2L1
```

```
cd pipeline/
python step1_get_img_features_new.py
```

## Step 2: post-processing of the single-camera tracklets
```
python step2_mtsc_post_process_new.py
```

## Step 3: multi-camera tracking
```
python step3_multi_camera_tracking_new.py
```

## Step 4: merge the results from each camera
```
python step4_merge_results_new.py
```

## Step 5: evaluate the results
**Evaluation for the first time:**
```
cd ../eval/
pip install -r requirement.txt
python group_gt.py
```

**Pedestrian Datasets**
```
source step5_eval_test.sh
```

**CityFlowV2 Validation Set**
Run one of the following commands to check the result of each scenario.
```
source step5_eval_validation.sh
source step5_eval_validation_S02.sh
source step5_eval_validation_S05.sh
```
"# MCMOT" 
