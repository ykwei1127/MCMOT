from .model import MCT
from .model_2 import MCT as MCT_2

def build_model(cfg, device):
    dim = cfg.MCT.FEATURE_DIM * 2
    model = MCT(dim, device)
    model = model.to(device)
    return model

def build_model_2(cfg, device):
    dim = cfg.MCT.FEATURE_DIM
    model = MCT_2(dim, device)
    model = model.to(device)
    return model