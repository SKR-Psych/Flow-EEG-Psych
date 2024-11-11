# replication/scripts/data_inspection.py

import sys
import os

# Add the path to the src/preprocess directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_dir = os.path.abspath(os.path.join(current_dir, '../../src/preprocess'))
sys.path.append(preprocess_dir)

from verify_mat_files import inspect_mat_file

def main():
    print("Starting data inspection using verify_mat_files.py...")
    inspect_mat_file()
    print("Data inspection completed.")

if __name__ == "__main__":
    main()
