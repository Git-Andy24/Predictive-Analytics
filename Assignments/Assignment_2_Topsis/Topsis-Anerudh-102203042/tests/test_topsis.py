import pytest
import pandas as pd
import subprocess

def test_topsis_cli():
    test_file = "test_data.csv"
    result_file = "test_result.csv"
    
    df = pd.DataFrame({
        "Fund Name": ["M1","M2","M3","M4","M5","M6","M7","M8"],
        "P1": [0.9,0.89,0.85,0.83,0.62,0.64,0.8,0.81],
        "P2": [0.81,0.79,0.72,0.69,0.38,0.41,0.64,0.66],
        "P3": [6.7,4.8,4.2,5,6.7,5.5,6.9,3.3],
        "P4": [32.3,62.1,42.5,30.5,62.8,58.2,57.5,54.5],
        "P5": [10.18,17.15,12.07,9.26,17.63,16.19,16.46,14.82]
    })
    df.to_csv(test_file, index=False)
    
    result = subprocess.run(
        ["topsis", test_file, "1,1,1,2,2", "+,+,-,-,+", result_file],
        capture_output=True,
        text=True
    )
    
    assert "TOPSIS analysis completed" in result.stdout
    
    result_df = pd.read_csv(result_file)
    assert "Topsis Score" in result_df.columns
    assert "Rank" in result_df.columns
