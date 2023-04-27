import nfl_data_py as nfl
import polars as pl
import pandas as pd
import numpy as np
from numba import jit
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

'''
Must analyze all data into rankings yearly between 1-32 all the following (x-axis)):
    - Points For (PF)
    - Points Differential (PD)
    - Off Points Rank (Off Pts Rank)
    - Off Yds Rank (Off Yds Rank)
    - Def Points Against Rank (Def Pts Rank)
    - Def Yds Against Rank (Def Yds Rank)
    - Turnover Differential (T/G)
    - Strength of Schedule (SOS)

Then, make this a graph of these stats ranked in terms of Wins (y-axis):
    - Wins (W)

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

# @jit(nopython=True)
def get_decade_list(dts_df):
    '''
    returns a np array of dataframes for each year in the decade
    '''
    decade_list = []
    for year in decade:
        temp_df = dts_df.filter(pl.col("Year") == year)
        year_df = temp_df.filter(pl.col("W") >= pl.col("L"))
        year_df = ( year_df.groupby("Tm").agg(
            [
                pl.col('W').sum().alias('Wins'),
                pl.col("L").sum().alias('Losses'),
                pl.col("PF").sum().alias('Points For'),
                pl.col("PD").sum().alias('Point Difference'),
            ]
            )
        )
        year_df.drop_in_place("Tm")
        year_df.to_pandas() 
        decade_list.append(year_df)
    # convert to numpy array
    #decade_list = np.array(decade_list)
    return decade_list

def main():
    '''
    dts_df = pl.read_csv("./decade_team_stats.csv")
    print(dts_df)
    '''
    dts_df = pl.read_csv("./data/decade_team_stats.csv")
    elo_df = pl.read_csv("./data/nfl_elo.csv")
    #print(dts_df)

    # in terms of ranks (lower number is better)
    decade_sum_stats = ( dts_df.groupby("Tm").agg(
        [
            pl.col('W').sum().alias('total_wins'),
            pl.col("L").sum().alias('total_losses'),
            pl.col("T").sum().alias('total_ties'),
            pl.col("PF").sum().alias('total_points_for')
        ]
    )
       )
    decade_list = get_decade_list(dts_df)
    print(decade_list[-1])

    # graph as TSNE
    tsne = TSNE(n_components=2, perplexity=2)

    tsne_results = tsne.fit_transform(decade_list[-2])

    plt.scatter(tsne_results[:,0], tsne_results[:,1])

    for i, team in enumerate(decade_list[-1].columns):
        plt.annotate(team, (tsne_results[i,0], tsne_results[i,1]))

    plt.show()


    print(current_time())
    return 0


if __name__ == "__main__":
    main()