from yacs.config import CfgNode as CN

_C = CN()

_C.CHALLENGE_DATA_DIR = ''
_C.DET_SOURCE_DIR = ''
_C.REID_MODEL = '../reid/reid_model/resnet101_ibn_a_2.pth'
_C.REID_MODEL_2 = '../reid/reid_model/resnet101_ibn_a_3.pth'
_C.REID_BACKBONE = 'resnet101_ibn_a'
_C.REID_MODEL_3 = '../reid/reid_model/resnext101_ibn_a_2.pth'
_C.REID_BACKBONE_3 = 'resnext101_ibn_a'
_C.REID_SIZE_TEST = [384, 384]
# _C.REID_SIZE_TEST = [256, 256]

_C.DET_IMG_DIR = ''
_C.DATA_DIR = ''
_C.ROI_DIR = ''
_C.CID_BIAS_DIR = ''

_C.USE_RERANK = False
_C.USE_FF = False
_C.SCORE_THR = 0.5

_C.MCMT_OUTPUT_TXT = ''
