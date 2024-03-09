import os
import glob
# PUB_WeeklyMarket_20240215
# Define the path to the data directory
data_directory = 'data/downloaded_csvs/'
# Get all csv files in the data directory
all_csv_files = glob.glob(data_directory + '*.csv')

# print(all_csv_files[0].split('_')[-1][:4])
# Filter out files from years before 2022
def filter_and_delete_versioned_files(all_csv_files):
    # Create a list of files with version numbers to delete
    files_to_delete = [file for file in all_csv_files if '_v' in file]
    
    # Delete the files with version numbers
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")

# Call the function to delete versioned files
# filter_and_delete_versioned_files(all_csv_files)

def delete():
    files_to_delete = []
    file_cap = None
    try:
        for file in all_csv_files:
            file_cap = file 
            if int(file.split('_')[-1][:4]) < 2022:
                files_to_delete.append(file)
    except Exception as e:
        print(e, file_cap)

    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")
# print(files_to_delete)
# exit()
for file_path in all_csv_files:
    with open(file_path, 'r') as file:
        lines = file.readlines()
    with open(file_path, 'w') as file:
        file.writelines(lines[4:])



# Create a list of all csv files ending with '_v1.csv' or '_v24.csv'
# files_to_delete = glob.glob(data_directory + '*_v1.csv') + glob.glob(data_directory + '*_v24.csv') + glob.glob(data_directory + '*_v2.csv')

# Iterate over the list of files and delete each one
