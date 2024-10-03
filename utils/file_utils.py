import h5py
import os
import torch
import numpy as np

def get_demo_data(dataset_path):
    
    f = h5py.File(dataset_path, "r")
    return f["data"]