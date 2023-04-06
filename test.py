import nfl_data_py as nfl
import polars as pl
import numpy as np
import os
from numba import jit
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE

start_time = datetime.now()

def current_time():
    return (datetime.now() - start_time).total_seconds()

def import_all_data():
    df = nfl.import_pbp_data([2021])
    df.to_csv("./data/pbp.csv")
    return df

def main():
    # Load the data
    #df = nfl.import_pbp_data([2021])
    #df.to_csv("./data/pbp.csv")
    if not os.path.exists("./data/pbp.csv"):
        df = import_all_data()
    
    pldf = pl.read_csv("./data/pbp.csv")
    print(pldf)
    print(current_time())
    return 0


if __name__ == "__main__":
    main()