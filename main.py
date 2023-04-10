import nfl_data_py as nfl
import polars as pl
import numpy as np
from numba import jit
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE

'''
dss_df: Decade Season Stats DataFrame from 2012-2021

'''

decade = [year for year in range(2012, 2022)]
start_time = datetime.now()

def current_time():
    return (datetime.now() - start_time).total_seconds()

@jit(nopython=True)
def analyze_dts(dts_np):
    '''
    dts_np: Decade Team Stats Numpy Array
    Utilize a numpy array to analyze the data of NFL teams over the past decade
    '''
    return 0

def main():
    '''
    dts_df = pl.read_csv("./decade_team_stats.csv")
    print(dts_df)
    '''
    dts_df = pl.read_csv("./data/decade_team_stats.csv")
    elo_df = pl.read_csv("./data/nfl_elo.csv")
    print(dts_df)
    print(elo_df)
    print(current_time())
    return 0


if __name__ == "__main__":
    main()