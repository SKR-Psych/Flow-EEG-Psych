import os
from load_data import load_mat_file, get_mat_file_list

def verify_mat_files():
    # Define paths relative to this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(current_dir, '../../data/raw/')

    # Get list of .mat files
    mat_files = get_mat_file_list(raw_data_dir)

    all_files_good = True

    for mat_file in mat_files:
        print(f'Verifying {mat_file}...')

        # Load the .mat file
        try:
            mat = load_mat_file(os.path.join(raw_data_dir, mat_file))

            # Check for 'BF_Data' key
            if 'BF_Data' not in mat:
                print(f'Error: {mat_file} does not contain BF_Data.')
                all_files_good = False
                continue

            BF_Data = mat['BF_Data']

            # Check for 'actualVariable' and 'EEG_full'
            if not hasattr(BF_Data, 'actualVariable') or not hasattr(BF_Data.actualVariable, 'EEG_full'):
                print(f'Error: {mat_file} does not have the expected structure (actualVariable.EEG_full).')
                all_files_good = False
                continue

            EEG_full = BF_Data.actualVariable.EEG_full

            # Verify data
            if not hasattr(EEG_full, 'data'):
                print(f'Error: {mat_file} EEG_full does not contain data.')
                all_files_good = False

            # Verify events
            if not hasattr(EEG_full, 'event'):
                print(f'Error: {mat_file} EEG_full does not contain events.')
                all_files_good = False

            # Verify channel locations
            if not hasattr(EEG_full, 'chanlocs'):
                print(f'Error: {mat_file} EEG_full does not contain chanlocs.')
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

