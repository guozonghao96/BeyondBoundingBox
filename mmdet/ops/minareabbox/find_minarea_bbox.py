import torch
from . import minareabbox_cuda

def find_minarea_rbbox(pred):
#     num_pred = pred.size(0)
    rbbox = minareabbox_cuda.minareabbox(pred)
    rbbox = rbbox.reshape(-1, 8)
    return rbbox