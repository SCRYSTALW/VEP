import os
import re
import numpy as np
import pandas as pd
from scipy.fftpack import fft
import glob
import shutil

# Directory paths
source_directory = '/Users/sc/Downloads/Dyslexia/VEP'
destination_directory = '/Users/sc/Downloads/Dyslexia/VEP_2'
#
# # Ensure the destination directory exists
# os.makedirs(destination_directory, exist_ok=True)
#
# # Regular expression pattern to match the relevant parts of the filename
# pattern = re.compile(r'(\w+)[ _]+[\w ]*[ _]+([vhVH]\d+\.?\d*)[cpd]*\.data')
#
# # Rename and copy files to the new directory
# for file_path in glob.glob(os.path.join(source_directory, '**', '*.data'), recursive=True):
#     file_name = os.path.basename(file_path)
#     match = pattern.match(file_name)
#
#     if match:
#         ID = match.group(1)  # e.g., DD003
#         orient_sf = match.group(2)  # v0.63 or h6.02
#         orient = orient_sf[0]  # 'v' or 'h'
#         sf = orient_sf[1:]  # Extract numerical part of spatial frequency, e.g., '6.02' or '0.63'
#         new_file_name = f"{ID}_{orient}_{sf}.data"
#         new_file_path = os.path.join(destination_directory, new_file_name)
#         shutil.copy(file_path, new_file_path)  # Copy the file to the new directory
#         print(f"Copied {file_name} to {new_file_name}")
#     else:
#         print(f"Skipping file with unexpected format: {file_name}")
file_list = glob.glob(os.path.join(destination_directory, '*.data'))

# Define the maxcal function
def maxcal(d):
    # D = d * 1E6  # change to uV
    D = d
    D = D[~np.isnan(D)]  # Remove NaN values
    # D = D[~np.isnan(d)]  # Remove NaN values

    if len(D) == 0:  # Check if the data is empty after removing NaNs
        return np.nan
    D_FFT = np.abs(fft(D))
    if len(D_FFT) == 0:  # Check if the FFT result is empty
        return np.nan
    # max_strength = np.max(D_FFT[:200])
    max_strength = np.max(D_FFT)

    return max_strength
# List to hold all results for final CSV
all_results = []

# Process each file individually
for file_path in file_list:
    try:
        # Load the data from the file
        data = np.genfromtxt(file_path, delimiter=',', encoding='ISO-8859-1')
        if data.ndim == 1:  # Ensure data is two-dimensional
            data = data.reshape(-1, 1)

        # Apply the maxcal function to the data
        msdata = np.apply_along_axis(maxcal, 0, data)

        # Debugging: Check the maxcal results
        print(f"Maxcal results from {file_path}:\n{msdata}")

        msdata = np.column_stack((msdata, np.zeros((len(msdata), 4))))

        # Create a DataFrame to store the results
        msdata_df = pd.DataFrame(msdata, columns=["maxstr", "ID", "orient", "SF", "trial"])

        # Split file name
        file_name = os.path.basename(file_path)
        split_data = file_name.split('_')

        if len(split_data) < 3:
            print(f"Skipping file due to unexpected filename format: {file_name}")
            continue

        ID = split_data[0]  # ND015
        orient = split_data[1].upper()  # V or H
        sf = split_data[2].split('.')[0]  # 0.63 or 6.02

        # Debugging statements
        print(f"Processing file: {file_name}")
        print(f"ID: {ID}, Orient: {orient}, SF: {sf}")

        # Update DataFrame with file-specific data
        msdata_df["ID"] = ID
        msdata_df["orient"] = orient
        msdata_df["SF"] = sf

        # Debugging: Check the DataFrame before aggregation
        print(f"msdata_df before aggregation:\n{msdata_df.head()}")

        # Convert maxstr and SF to numeric and group by ID, orient, SF
        msdata_df["maxstr"] = pd.to_numeric(msdata_df["maxstr"], errors='coerce')
        msdata_df["SF"] = pd.to_numeric(msdata_df["SF"], errors='coerce')

        # Compute mean by condition
        msdata1 = msdata_df.groupby(["ID", "orient", "SF"]).agg(
            {"maxstr": lambda x: np.exp(np.mean(np.log(x[x > 0])))}).reset_index()

        # Debugging: Check the aggregated DataFrame
        print(f"msdata1 after aggregation:\n{msdata1.head()}")

        # # Rename SF
        # sf_mapping = {"0.63": 0.63, "6.02": 6.02}
        # msdata1["SF"] = msdata1["SF"].astype(str).map(sf_mapping)
        # msdata1 = msdata1.sort_values(by=["ID", "SF"])

        # # Debugging: Check the DataFrame before pivot
        # print(f"msdata1 before pivot:\n{msdata1.head()}")
        #
        # # Pivot wider
        # myresults = msdata1.pivot_table(index="ID", columns=["orient", "SF"], values="maxstr").reset_index()
        #
        # # Debugging: Check the pivoted DataFrame
        # print(f"myresults after pivot:\n{myresults.head()}")

        # Append results to the list
        all_results.append(msdata1)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

# Concatenate all results into a single DataFrame
if all_results:
    final_results = pd.concat(all_results, ignore_index=True)

    # Save all results to a single CSV file
    output_file = "/Users/sc/Downloads/Dyslexia/all_maresults.csv"
    final_results.to_csv(output_file, index=False)

    print(f"Results saved to {output_file}")
else:
    print("No results to save.")