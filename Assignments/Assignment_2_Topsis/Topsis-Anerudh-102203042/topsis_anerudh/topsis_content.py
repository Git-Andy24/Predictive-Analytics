import pandas as pd
import numpy as np
import sys

def validate_inputs(df, weights, impacts):
    if df.shape[1] < 3:
        raise ValueError("Input file must contain at least three columns.")
    
    try:
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float)
    except ValueError:
        raise ValueError("All columns except the first must contain numeric values only.")
    
    weights = list(map(float, weights.split(',')))
    impacts = impacts.split(',')
    
    if len(weights) != df.shape[1] - 1 or len(impacts) != df.shape[1] - 1:
        raise ValueError("The number of weights and impacts must match the number of numeric columns in the dataset.")
    
    if not all(i in ['+', '-'] for i in impacts):
        raise ValueError("Impacts must be either '+' or '-'.")
    
    return df, weights, impacts

def topsis(df, weights, impacts):
    data = df.iloc[:, 1:].values.astype(float)
    weights = np.array(weights)
    
    norm_matrix = data / np.sqrt((data**2).sum(axis=0))
    weighted_matrix = norm_matrix * weights
    
    ideal_best = np.zeros(weighted_matrix.shape[1])
    ideal_worst = np.zeros(weighted_matrix.shape[1])

    for i in range(weighted_matrix.shape[1]):
        if impacts[i] == '+':
            ideal_best[i] = weighted_matrix[:, i].max()
            ideal_worst[i] = weighted_matrix[:, i].min()
        else:  #negative impact
            ideal_best[i] = weighted_matrix[:, i].min()
            ideal_worst[i] = weighted_matrix[:, i].max()
    
    distance_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))

    scores = distance_worst / (distance_best + distance_worst)

    df['Topsis Score'] = scores
    df['Rank'] = df['Topsis Score'].rank(ascending=False, method='dense').astype(int)
    
    return df

def run_topsis(input_file, weights, impacts, result_file):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        sys.exit(f"Error: File '{input_file}' not found.")
    
    df, weights, impacts = validate_inputs(df, weights, impacts)
    result_df = topsis(df, weights, impacts)
    result_df.to_csv(result_file, index=False)
    print(f"TOPSIS analysis completed. Results saved to {result_file}.")
