import argparse
import os
import numpy as np
import pandas as pd
import sys
import plotly.graph_objects as go
import math
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def main(args):
    # Access the arguments
    threshold_year = int(args.threshold_year)
    threshold_z_score = float(args.threshold_z_score)
    
    # Print the arguments
    print(f"\t ****Threshold Year: {threshold_year}\n\tThis year is used to split the time series into 2 periods")
    factors_TS_augmented_directory= 'factors_TS_augmented'
    factors_TS_directory = 'factors_TS'
    op_directory = 'distributions'

    files_augmented = os.listdir(factors_TS_augmented_directory)
    files_orig = os.listdir(factors_TS_directory)
    # Filter CSV files
    csv_files_augmented = [file for file in files_augmented if file.endswith('.csv')]
    if len(csv_files_augmented) == 0:
        print(f"----------  No CSV files found in {factors_TS_augmented_directory}. Exting now  ----------")
        sys.exit(1)
    csv_files_orig = [file for file in files_orig if file.endswith('.csv')]
    if len(csv_files_orig) == 0:
        print(f"----------  No CSV files found in {factors_TS_directory}. Exting now  ----------")
        sys.exit(1)

    # Loop over each CSV file
    for file_name in csv_files_augmented:
        # Load data from file
        aug_data = pd.read_csv(factors_TS_augmented_directory+'/'+file_name, header=0)
        min_val = min(aug_data['value'])
        max_val = max(aug_data['value'])
        
        # Rescale the normalized data to get the original time series
        for i in aug_data.columns[2:]:
            aug_data[i] = (aug_data[i] * (max_val - min_val)) + min_val
        aug_data.to_csv("Aug_data.csv", index=False)

        # import pdb
        # pdb.set_trace()
        print(f"\n----------- Loaded {file_name} -----------\n")
        #aug_data_abs = aug_data_abs.T
        aug_data=aug_data.reset_index(drop=True)
        # new_columns = [f'{file_name[:-4]}_AUG_{i}' for i in range(1, len(aug_data_abs.columns) + 1)]
        # aug_data_abs.columns = new_columns
        pct_col_list = []
        gdp_gr_col_list = []
        for i in aug_data.columns[2:]:
            new_col_name = f"Growth Rate for {i}"
            gdp_gr_col_list.append(new_col_name)
            new_col = aug_data[i].pct_change() * 100
            new_col = new_col.rename(new_col_name)
            pct_col_list.append(new_col)

        # Concatenate the new columns to the DataFrame
        aug_data = pd.concat([aug_data] + pct_col_list, axis=1)
        #aug_data[new_col] = aug_data[i].pct_change() * 100
        # aug_data_abs['date'] = orig_gdp_data['date']

        aug_data['date']=pd.to_datetime(aug_data['date'])

        outliers_all = []
        num_of_outliers = []
        for num, i in enumerate(gdp_gr_col_list):
            # if num>5:
            #     break
            # print(i)
            # Split the data into two dataframes, before and after threshold_year
            gdp_data_before_threshold_year = aug_data.loc[aug_data['date'].dt.year <= threshold_year, [i, 'date']]
            gdp_data_after_threshold_year = aug_data.loc[aug_data['date'].dt.year > threshold_year, [i, "date"]]

            # Calculate the mean and standard deviation of 'GDP Growth Rate' before threshold_year
            mean_before_threshold_year = gdp_data_before_threshold_year[i].mean()
            std_dev_before_threshold_year = gdp_data_before_threshold_year[i].std()

            # Calculate the z-score for each data point before threshold_year
            z_score_new_col = f"Z Score: {i}"
            gdp_data_before_threshold_year[z_score_new_col] = (gdp_data_before_threshold_year[i] - mean_before_threshold_year) / std_dev_before_threshold_year

            # Find outliers before threshold_year
            outliers_before_threshold_year = gdp_data_before_threshold_year[gdp_data_before_threshold_year[z_score_new_col] < -threshold_z_score]
            #print(f"outliers_before_threshold_year = {outliers_before_threshold_year}")
            #print(f"gdp_data_before_threshold_year[z_score_new_col] = {gdp_data_before_threshold_year[z_score_new_col]}")
            # print(f"Min z score:: {min(gdp_data_before_threshold_year[z_score_new_col])}")
            # Count the number of NaN values
            nan_count = gdp_data_before_threshold_year[z_score_new_col].isnull().sum()

            #print("Number of NaN values:", nan_count)
            # Find the index of the first NaN value
            nan_index = gdp_data_before_threshold_year[z_score_new_col].index[gdp_data_before_threshold_year[z_score_new_col].isnull()].tolist()

            #print("Index of first NaN value:", nan_index)
            # Calculate the mean and standard deviation of 'GDP Growth Rate' after threshold_year
            mean_after_threshold_year = gdp_data_after_threshold_year[i].mean()
            std_dev_after_threshold_year = gdp_data_after_threshold_year[i].std()

            # Calculate the z-score for each data point after threshold_year
            gdp_data_after_threshold_year[z_score_new_col] = (gdp_data_after_threshold_year[i] - mean_after_threshold_year) / std_dev_after_threshold_year

            # Find outliers after threshold_year (assuming threshold = 2)
            outliers_after_threshold_year = gdp_data_after_threshold_year[gdp_data_after_threshold_year[z_score_new_col] < -threshold_z_score]

            # Combine the outliers from both periods
            outliers = pd.concat([outliers_before_threshold_year, outliers_after_threshold_year])
            
            for index, row in outliers.iterrows():
                # Find the date just before the date in outliers
                date_before_outlier = aug_data.loc[aug_data['date'] < row['date'], 'date'].max()
                # Find the corresponding GDP Growth Rate in df
                if not pd.isnull(date_before_outlier):
                    #value_before_outlier = df.loc[df['date'] == date_before_outlier, i].values[0]
                    gdp_val_pos = aug_data.loc[aug_data['date'] == date_before_outlier, i].values[0]
                    if math.isnan(gdp_val_pos):
                        print(f"NaN at entry {num}")
                    else:
                        outliers_all.append(gdp_val_pos)
                        outliers_all.append(row[i])
                

                    
            # Print or further process the outliers
            #print(outliers)
            num_of_outliers.append(len(outliers))
        print(f"Max Outliers - {file_name[:-4]} = {max(num_of_outliers)}")
        print(f"Min Outliers - {file_name[:-4]} = {min(num_of_outliers)}")
        print(f"Avg Outliers - {file_name[:-4]} per TS= {sum(num_of_outliers)/len(num_of_outliers)*1.0}")

        print(f"Total number of data points for PDF for {file_name[:-4]} = {len(outliers_all)}" )
        # Create a kernel density estimate (KDE) for the data
        # import pdb

        # pdb.set_trace()

        if not os.path.exists(op_directory):
            os.makedirs(op_directory)
            print(f"Directory '{op_directory}' created.")
        else:
            print(f"Directory '{op_directory}' already exists.")

        if len(outliers_all) == 0:
            print(f"No Outliers for {file_name[:-4]}. z-score threshold = {threshold_z_score} is too high...")
            continue

        kde = gaussian_kde(outliers_all)
        # Generate x values for the plot
        x = np.linspace(min(outliers_all), max(outliers_all), 1000)

        # Calculate the PDF values using the KDE
        pdf_values = kde(x)

        # Find peaks in the data
        peaks, _ = find_peaks(pdf_values)

        # Create a Plotly figure
        fig = go.Figure()

        # Add a trace for the PDF
        fig.add_trace(go.Scatter(x=x, y=pdf_values, mode='lines', name='PDF'))
        # Add markers for the peaks
        fig.add_trace(go.Scatter(x=x[peaks], y=pdf_values[peaks], mode='markers', marker=dict(color='red'), name='Peaks'))

        # Update the layout
        title = f"PDF for {file_name[:-4]}, Z-score threshold: {threshold_z_score}"
        fig.update_layout(title=title,
                        xaxis_title='Value',
                        yaxis_title='Density')

        # Show the plot
        fig.show()

        # Save the plot as an HTML file
        html_file_name = f"PDF_{file_name[:-4]}_threshold_{threshold_z_score}.html"
        file_path = os.path.join(op_directory, html_file_name)
        fig.write_html(file_path)




    
    print(f"\n----------  Data fetched and saved successfully in {op_directory}  ----------\n")


if __name__ == "__main__":
    print("\n----------  Running outlier_distribution.py ----------\n")
    # Create argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add arguments with default values
    parser.add_argument('--threshold_year', default='1980', help='Threshold for year . Intervals denoting similar financial conditions')
    parser.add_argument('--threshold_z_score', default='2', help='Threshold for z-score. Outliers selected with z-score greater than threshold')
    
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args)
