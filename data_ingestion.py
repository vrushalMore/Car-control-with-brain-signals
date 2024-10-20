import os

def checkCsv(file_path):
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return False
    if file_path.endswith('.csv'):
        print(f"{file_path} is a CSV file.")
        return True
    else:
        print(f"{file_path} is not a CSV file.")
        return False

file_path = "eeg_data.csv"
checkCsv(file_path)
#Set standards