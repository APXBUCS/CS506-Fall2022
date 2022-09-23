import pandas as pd
def read_csv(csv_file_path):
    """
        Given a path to a csv file, return a matrix (list of lists)
        in row major.
    """
    print(pd.read_csv(csv_file_path, header=None).values)
    return (pd.read_csv(csv_file_path, header=None).values.tolist())

