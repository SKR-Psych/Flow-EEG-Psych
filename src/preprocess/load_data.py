# src/preprocess/load_data.py

import mat73
import os

def load_mat_file(file_path):
    mat = mat73.loadmat(file_path)
    return mat

def get_mat_file_list(raw_data_dir):
    mat_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.mat')]
    return mat_files


