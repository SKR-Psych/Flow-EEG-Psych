import h5py
import os

def load_mat_file(file_path):
    """
    Loads a MATLAB v7.3 .mat file using h5py and returns the content.
    """
    mat = h5py.File(file_path, 'r')
    return mat

def get_mat_file_list(raw_data_dir):
    """
    Returns a list of .mat files in the specified directory.
    """
    mat_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.mat')]
    return mat_files


