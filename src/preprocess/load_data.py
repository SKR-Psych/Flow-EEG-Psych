import scipy.io
import os

def load_mat_file(file_path):
    """
    Loads a .mat file and returns the content.
    """
    mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return mat

def get_mat_file_list(raw_data_dir):
    """
    Returns a list of .mat files in the specified directory.
    """
    mat_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.mat')]
    return mat_files

