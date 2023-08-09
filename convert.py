import pandas as pd
from scipy.io import arff

def arff_to_csv(arff_file, csv_file):
    # Load ARFF file
    data, meta = arff.loadarff(arff_file)

    # Convert ARFF data to a DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    arff_file_path = "cm1.arff"
    csv_file_path = "C:\\Users\\12345\\OneDrive\\Desktop\\AI_MP\\output_file.csv"
    arff_to_csv(arff_file_path, csv_file_path)
