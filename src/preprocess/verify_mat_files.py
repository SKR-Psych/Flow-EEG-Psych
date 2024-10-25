# src/preprocess/verify_mat_files.py

import os
from load_data import load_mat_file, get_mat_file_list

def inspect_mat_file():
    print("Starting inspection...")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        raw_data_dir = os.path.join(current_dir, '../../data/raw/')
        mat_files = get_mat_file_list(raw_data_dir)

        if not mat_files:
            print("No .mat files found in the raw data directory.")
            return

        # Inspect the first .mat file
        mat_file = mat_files[0]
        print(f'Inspecting {mat_file}...')
        mat = load_mat_file(os.path.join(raw_data_dir, mat_file))

        # Print the keys of the loaded .mat file
        print('Variables in the .mat file:')
        print(mat.keys())

        # Access 'actualVariable'
        if 'actualVariable' in mat:
            actual_variable = mat['actualVariable']
            print('\nType of actualVariable:', type(actual_variable))

            # If it's a dictionary, print its keys
            if isinstance(actual_variable, dict):
                print('Keys in actualVariable:')
                print(actual_variable.keys())

                # Further inspect the contents of 'actualVariable'
                for key in actual_variable.keys():
                    print(f"\nContents of '{key}':")
                    print(actual_variable[key])
            else:
                print('actualVariable is not a dict, printing attributes:')
                print(dir(actual_variable))

                if hasattr(actual_variable, '__dict__'):
                    actual_variable_dict = vars(actual_variable)
                    print('Keys in actualVariable:')
                    print(actual_variable_dict.keys())
                else:
                    print('Cannot access contents of actualVariable.')
        else:
            print("The .mat file does not contain 'actualVariable'.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    inspect_mat_file()




