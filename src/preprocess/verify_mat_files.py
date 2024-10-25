import os
from load_data import load_mat_file, get_mat_file_list

def verify_mat_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '../../data/raw/')
    mat_files = get_mat_file_list(raw_data_dir)
    all_files_good = True

    for mat_file in mat_files:
        print(f'Verifying {mat_file}...')

        try:
            mat = load_mat_file(os.path.join(raw_data_dir, mat_file))

            # Check for 'BF_Data'
            if 'BF_Data' not in mat:
                print(f'Error: {mat_file} does not contain BF_Data.')
                all_files_good = False
                continue

            # Check structure
            EEG_full = mat['BF_Data']['actualVariable']['EEG_full']
            if 'data' not in EEG_full or 'event' not in EEG_full or 'chanlocs' not in EEG_full:
                print(f'Error: {mat_file} EEG_full is missing required datasets.')
                all_files_good = False

        except Exception as e:
            print(f'Error verifying {mat_file}: {e}')
            all_files_good = False
            continue

    if all_files_good:
        print('All .mat files are verified and usable.')
    else:
        print('Some .mat files have issues. Please check the error messages above.')

if __name__ == '__main__':
    verify_mat_files()


