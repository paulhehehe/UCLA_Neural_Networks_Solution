import pandas as pd
import logging
import os

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)

def load_and_preprocess_data(data_path):
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            logging.error(f"File not found: {data_path}")
            return None
        
        # Import the data from CSV file
        logging.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        # Log success message with data shape
        logging.info(f"Data loaded successfully with shape: {df.shape}")

        df['Admit_Chance']=(df['Admit_Chance'] >=0.8).astype(int)
        df = df.drop(['Serial_No'], axis=1)
            
 
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None
    
