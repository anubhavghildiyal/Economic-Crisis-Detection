import argparse
import datetime
from full_fred.fred import Fred
import yaml
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd

def main(args):
    # Access the arguments
    start_date = args.start_date
    end_date = args.end_date
    yaml_file = args.yaml
    
    # Print the arguments
    print(f"\t****Start date: {start_date}")
    print(f"\t****End date: {end_date}")
    print(f"\t****YAML file: {yaml_file}")
    
    # Load environment variables from .env file
    load_dotenv()
    # Get the API key from the environment
    api_key = os.getenv("FRED_API_KEY_FILE")
    # Create an instance of Fred with the API key
    fred = Fred(api_key)
    fred.get_api_key_file()

    # Read the factors from the YAML file
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    # Access the list of indicators
    indicators = config.get('factors', [])

    # Directory path
    factor_TS_directory = 'factors_TS'
    # Create the directory if it does not exist
    if not os.path.exists(factor_TS_directory):
        os.makedirs(factor_TS_directory)
        print(f"Directory '{factor_TS_directory}' created.")
    else:
        print(f"Directory '{factor_TS_directory}' already exists.")

    # Fetch data for each indicator
    print(f"\n----------  Fetching data for {len(indicators)} indicators  ----------")
    print(f"----------  Indicators:  ----------\n")
    data = {}
    for indicator in indicators:
        name = indicator.get('name', '')
        code = indicator.get('code', '')
        print(f"- {name}: {code}")
        series_data = fred.get_series_df(code, observation_start=start_date, observation_end=end_date)
        data = series_data['value'].values
        # Normalize data
        try:
            # Try to convert data to float
            data = data.astype(float)
        except ValueError as e:
            # Handle the error (e.g., print a message, fill missing values, etc.)
            print(f"Error converting data to float: {e}")
            # Convert 'value' column to numeric, coerce errors to NaN
            series_data['value'] = pd.to_numeric(series_data['value'], errors='coerce')
            # Remove rows where 'value' column contains NaN (i.e., non-numeric values)
            series_data = series_data.dropna(subset=['value']).reset_index(drop=True)
            data = series_data['value'].values.astype(float)

        # Normalize data
        data_min = np.min(data)
        data_max = np.max(data)
        data_normalized = (data - data_min) / (data_max - data_min)
        series_data['Normalised Value'] = data_normalized

        print(f"Writing to... {factor_TS_directory}/{name}_{code}.csv")
        series_data.to_csv(factor_TS_directory+'/'+name+'_'+code+".csv", index=False)

    print(f"\n----------  Data fetched and saved successfully in {factor_TS_directory} ----------\n")


if __name__ == "__main__":
    print("\n---------- Get Time Series from FRED ----------\n")
    # Get today's date for the end date
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    
    # Add arguments with default values
    parser.add_argument('--start-date', default='1913-01-01', help='Start date of the time series')
    parser.add_argument('--end-date', default=end_date, help='End date of the time series')
    parser.add_argument('--yaml', default='factors.yaml', help='Provide a yaml file')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
