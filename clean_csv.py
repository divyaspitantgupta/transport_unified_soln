import pandas as pd
import re
import os

def is_valid_plate(text):
    """Regex check for Indian License Plates"""
    if pd.isna(text) or str(text).strip() == "N/A":
        return False
        
    clean_text = str(text).replace(" ", "").replace("-", "").upper()
    # Pattern: 2 Letters, 1-2 Numbers, 1-3 Letters, 4 Numbers
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$'
    return bool(re.match(pattern, clean_text))

def clean_traffic_data(input_csv, output_csv):
    if not os.path.exists(input_csv):
        print(f"Error: '{input_csv}' not found. Please run main.py first to generate data.")
        return

    try:
        # 1. Raw CSV file load karo
        df = pd.read_csv(input_csv, encoding='latin1')
        
        # 2. Check if file is empty or missing columns
        if df.empty or 'License_Plate' not in df.columns:
            print("CSV is empty or missing 'License_Plate' column.")
            return

        # 3. Regex filter apply karo aur ek naya column banao
        df['Is_Valid'] = df['License_Plate'].apply(is_valid_plate)
        
        # 4. Sirf True (Valid) plates wale rows ko rakho
        clean_df = df[df['Is_Valid'] == True]
        
        # 5. Extra 'Is_Valid' column hata do aur naya CSV save karo
        clean_df = clean_df.drop(columns=['Is_Valid'])
        clean_df.to_csv(output_csv, index=False)
        
        print("--- Data Cleaning Summary ---")
        print(f"Total raw entries: {len(df)}")
        print(f"Valid entries kept: {len(clean_df)}")
        print(f"Garbage entries removed: {len(df) - len(clean_df)}")
        print(f"Cleaned file saved to: {output_csv}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Input aur Output files ke naam
    RAW_CSV = "speed_violations.csv"
    CLEAN_CSV = "clean_speed_violations.csv"
    
    clean_traffic_data(RAW_CSV, CLEAN_CSV)