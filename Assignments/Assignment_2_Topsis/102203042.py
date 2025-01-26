import sys
import pandas as pd
import numpy as np

def validate_inputs(input_file, weights, impacts, result_file):
    try:
        df = pd.read_csv(input_file)
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unable to read '{input_file}'. {e}")
        sys.exit(1)
    
    if df.shape[1] < 3:
        print("Error: Input file must contain at least three columns.")
        sys.exit(1)
    

    try:
        df.iloc[1:, 1:] = df.iloc[1:, 1:].astype(float)
    except ValueError:
        print("Error: All columns except the first must contain numeric values only.")
        sys.exit(1)
    
    weights = list(map(float, weights.split(',')))
    impacts = impacts.split(',')
    
    if len(weights) != df.shape[1] - 1 or len(impacts) != df.shape[1] - 1:
        print(len(weights)-(df.shape[1]-1)," ",len(impacts)-(df.shape[1] - 1))
        print("Error: The number of weights and impacts must match the number of numeric columns in the dataset.")
        sys.exit(1)
    
    if not all(i in ['+', '-'] for i in impacts):
        print("Error: Impacts must be either '+' or '-'.")
        sys.exit(1)
    
    return df, weights, impacts

def topsis(df, weights, impacts):
    data = df.iloc[:, 1:].values.astype(float)
    weights = np.array(weights)
    
    norm_matrix = data / np.sqrt((data**2).sum(axis=0)) #normalising matrix
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
    
    scores = distance_worst / (distance_best + distance_worst) #topsis score
    
    df['Topsis Score'] = scores
    df['Rank'] = df['Topsis Score'].rank(ascending=False, method='dense').astype(int)
    
    return df

def main():
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)
    
    input_file, weights, impacts, result_file = sys.argv[1:]
    df, weights, impacts = validate_inputs(input_file, weights, impacts, result_file)
    result_df = topsis(df, weights, impacts)
    # print(result_df)
    result_df.to_csv(result_file, index=False)
    print(f"TOPSIS analysis completed. Results saved to {result_file}.")

if __name__ == "__main__":
    main()
