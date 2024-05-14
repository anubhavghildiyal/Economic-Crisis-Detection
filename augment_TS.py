import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.losses import mse, binary_crossentropy # type: ignore
import pandas as pd
from scipy.stats import pearsonr
#import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.models import load_model
from keras.models import save_model
from statistics import median
import sys
import pdb
import json
from datetime import datetime
import random

def main(args):
    
    # Access the arguments
    number = int(args.number)
    train = args.train
    min_corr = float(args.min_corr)

    factor_TS_directory= 'factors_TS'
    if not os.path.exists(factor_TS_directory):
        os.makedirs(factor_TS_directory)
        print(f"Directory '{factor_TS_directory}' created.")
    else:
        print(f"Directory '{factor_TS_directory}' already exists.")
    # Get the current time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Print the arguments
    print(f"Number of augmented time series: {number}")
    if train:
        # Directory path
        op_directory = 'autoencoders'
        # Create the directory if it does not exist
        if not os.path.exists(op_directory):
            os.makedirs(op_directory)

        # List all files in the raw TS directory
        files = os.listdir(factor_TS_directory)
        # Filter CSV files
        csv_files = [file for file in files if file.endswith('.csv')]
        if len(csv_files) == 0:
            print(f"----------  No CSV files found in {factor_TS_directory}. Exting now  ----------")
            sys.exit(1)
        # Define the VAE architecture
        latent_dim = 2
        # Loop over each CSV file
        for file_name in csv_files:
            # Load data from file
            df = pd.read_csv(factor_TS_directory+'/'+file_name, header=0)
            print(f"----------- Loaded {file_name} -----------\n")
            # Extract the 'value' column as input data
            data_normalized = df['Normalised Value'].values

            # Reshape input data to have the correct shape
            num_samples = len(data_normalized)
            num_features = 1  # Assuming each data point is a single feature
            data_normalized = data_normalized.reshape(num_samples, num_features)
            
            # Define the input shape
            input_shape = data_normalized.shape[1]

            # Define the autoencoder architecture
            input_data = Input(shape=(input_shape,))
            encoded = Dense(64, activation='relu')(input_data)
            encoded = Dense(32, activation='relu')(encoded)
            encoded = Dense(16, activation='relu')(encoded)
            decoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(64, activation='relu')(decoded)
            decoded = Dense(input_shape, activation='sigmoid')(decoded)  # Reconstruct original input shape

            # Create the autoencoder model
            autoencoder = Model(input_data, decoded)
            # Compile the model
            autoencoder.compile(optimizer='adam', loss='mse')
            # Train the autoencoder
            autoencoder.fit(data_normalized, data_normalized, epochs=100, batch_size=32)

            # Save the model with a specific name format
            model_name = f'{op_directory}/{file_name[:-4]}.keras'  # Replace <name> and <code> with actual values
            autoencoder.save(model_name)
    
    # Directory path
    directory = 'autoencoders'
    # List all files in the directory
    files = os.listdir(directory)
    # Filter .keras model files
    model_files = [file for file in files if file.endswith('.keras')]
    if len(model_files) == 0:
        print(f"----------  No model files found in the {directory}. Exting now  ----------")
        sys.exit(1)
    # Directory path
    op_directory = 'factors_TS_augmented'
    # Create the directory if it does not exist
    if not os.path.exists(op_directory):
        os.makedirs(op_directory)
    # Loop over each .keras model file
    for model_file in model_files:
        print(f"\n------------ Loading model from file: {model_file} ------------\n")
        # Load the model
        model = load_model(os.path.join(directory, model_file))
        df = pd.read_csv(factor_TS_directory+'/'+model_file[:-6]+'.csv', header=0)
        data_normalized = df['Normalised Value'].values
        # Reshape input data to have the correct shape
        num_samples = len(data_normalized)
        num_features = 1  # Assuming each data point is a single feature
        data_normalized = data_normalized.reshape(num_samples, num_features)


        # # Save the array to a text file
        # np.savetxt('data_nomralized_augment_ts.txt', data_normalized.flatten())

        # # Read the array back from the text file
        # loaded_arr = np.loadtxt('data_nomralized_data_aug_viz.txt')

        # # print("Original array:")
        # # print(data_normalized.flatten())

        # # print("\nLoaded array:")
        # # print(loaded_arr)
        # if np.array_equal(data_normalized.flatten(), loaded_arr):
        #     print("Arrays are equal.")
        # else:
        #     print("Arrays are not equal.")

        # Initialize arrays to store correlations and noisy data
        correlations = []
        noisy_data_list = []
        enc_op_to_noise_list = []
        generated_ts = 0
        #noisy_data_list.append(list(df['date']))
        while generated_ts < number:
            # #Generate noisy data
            # if _%6==0:
            #     noise = np.random.normal(loc=2, scale=0.01, size=data_normalized.shape)
            # elif _%6==1:
            #     noise = np.random.normal(loc=2.2, scale=0.02, size=data_normalized.shape)
            # elif _%6==2:
            #     noise = np.random.normal(loc=2.4, scale=0.03, size=data_normalized.shape)
            # elif _%6==3:
            #     noise = np.random.normal(loc=2.6, scale=0.02, size=data_normalized.shape)
            # elif _%6==4:
            #     noise = np.random.normal(loc=2.3, scale=0.01, size=data_normalized.shape)
            # elif _%6==5:
            #     noise = np.random.normal(loc=2.1, scale=0.04, size=data_normalized.shape)
            # else:
            #     noise = np.random.normal(loc=1.5, scale=0.04, size=data_normalized.shape)

            _ = random.randint(1, 6)
            if _%6==0:
                noise = np.random.normal(loc=0, scale=0.015, size=data_normalized.shape)
            elif _%6==1:
                noise = np.random.normal(loc=0, scale=0.023, size=data_normalized.shape)
            elif _%6==2:
                noise = np.random.normal(loc=0.3, scale=0.02, size=data_normalized.shape)
            elif _%6==3:
                noise = np.random.normal(loc=-0.3, scale=0.02, size=data_normalized.shape)
            elif _%6==4:
                noise = np.random.normal(loc=0, scale=0.01, size=data_normalized.shape)
            elif _%6==5:
                noise = np.random.normal(loc=-0.2, scale=0.04, size=data_normalized.shape)
            else:
                noise = np.random.normal(loc=0.2, scale=0.04, size=data_normalized.shape)
            
            
            # _ = random.randint(1, 6)
            # #Generate noisy data
            # if _%5==0:
            #     noise = np.random.normal(loc=2, scale=0.01, size=data_normalized.shape)
            # elif _%5==1:
            #     noise = np.random.normal(loc=2.2, scale=0.02, size=data_normalized.shape)
            # elif _%5==2:
            #     noise = np.random.normal(loc=2.4, scale=0.03, size=data_normalized.shape)
            # elif _%5==3:
            #     noise = np.random.normal(loc=2.5, scale=0.01, size=data_normalized.shape)
            # else:
            #     noise = np.random.normal(loc=1.9, scale=0.04, size=data_normalized.shape)
            
            #noise = np.random.normal(loc=0, scale=0.04, size=data_normalized.shape)
            noisy_data = data_normalized + noise
            noisy_data_list.append(noisy_data.flatten())
            #print(f"noisy data = {noisy_data}")
            # Predict output
            predicted_output = model.predict(noisy_data)
            
    
            # Calculate correlation
            correlation, _ = pearsonr(data_normalized.flatten(), predicted_output.flatten())
            #correlation, _ = pearsonr(data_normalized.flatten(), noisy_data.flatten())
            correlations.append(correlation)
            if correlation > min_corr:
                generated_ts += 1
                print('---------------------- yessss')
                enc_op_to_noise_list.append(predicted_output.flatten())
            
        # Convert noisy_data_list to DataFrame
        #noisy_data_df = pd.DataFrame(np.array(noisy_data_list).reshape(-1, number) )
        noisy_data_df = pd.DataFrame(noisy_data_list).T
        enc_op_to_noise_df = pd.DataFrame(enc_op_to_noise_list).T
        #if enc_op_to_noise_df.shape[1]
        # import pdb

        # pdb.set_trace()

        # Save noisy data to a file
        op_file_name = f'{op_directory}/{model_file[:-6]}.csv'
        #noisy_data_df.reset_index(drop=True, inplace=True)
        #new_columns = [f'{model_file[:-6]}_AUG_{i}' for i in range(1, len(noisy_data_df.columns) + 1)]
        
        new_columns = [f'{model_file[:-6]}_AUG_{i}' for i in range(1, len(enc_op_to_noise_df.columns) + 1)]
        #new_columns2 = [f'{model_file[:-6]}_AUG_noise_{i}' for i in range(1, len(noisy_data_df.columns) + 1)]

        # noisy_data_df.columns = new_columns
        # noisy_data_df = pd.concat([df['date'], noisy_data_df], axis=1)
        # noisy_data_df = pd.concat([df['value'], noisy_data_df], axis=1)
        # noisy_data_df.to_csv(op_file_name, mode='w', index=False)

        enc_op_to_noise_df.columns = new_columns
        enc_op_to_noise_df = pd.concat([df['date'], enc_op_to_noise_df], axis=1)
        enc_op_to_noise_df = pd.concat([df['value'], enc_op_to_noise_df], axis=1)
        #enc_op_to_noise_df = pd.concat([enc_op_to_noise_df, noisy_data_df], axis=1)

        enc_op_to_noise_df.to_csv(op_file_name, mode='w', index=False)

        # Create a dictionary with the correlation statistics
        output_dict = {
            "timestamp": current_time,
            "op_file_name": op_file_name,
            "model_file": model_file[:-6],
            "min_correlation": min(correlations),
            "max_correlation": max(correlations),
            "median_correlation": median(correlations),
        }

        # Define the JSON file path
        json_file_path = f"{op_directory}/correlation_statistics.json"

        # Write the dictionary to the JSON file
        with open(json_file_path, "a") as json_file:
            json.dump(output_dict, json_file, indent=4)

        print(f"\n---------- Data augmented and saved to: {op_file_name} ----------")
        print(f'---------- Correlation statistics for {model_file[:-6]} ----------\n')
        print(f"\t\t**** Min correlation: {min(correlations)}")
        print(f"\t\t**** Max correlation: {max(correlations)}")
        print(f"\t\t**** Median correlation: {median(correlations)}")


if __name__ == "__main__":
    print("\n---------- Augment All Time Series data  ----------\n")
    # Create argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    # Add arguments with default values
    parser.add_argument('--number', default=1000, help='Number of augmented time series')
    parser.add_argument('--train', default=True, help='Specify if autoencoder should be trained, otherwise it will be picked from ./autencoders directory')
    parser.add_argument('--min_corr', default=0.95, help='Minimum correlation between original and augmented time series')
    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with parsed arguments
    main(args)
