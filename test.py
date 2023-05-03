import nfl_data_py as nfl
import numpy as np
import pandas as pd
import polars as pl

decade = [year for year in range(2012, 2021)]

'''
Using the ELO rating system, predict the outcome of each game in the 2022 season
'''


def get_predictions():
    '''
    Perform ELO calculations for each team in the decade
    '''
    
    return 

def main():
    '''
    week_data = nfl.import_weekly_data(decade)
    week_data.to_csv("./data/weekly_data.csv")
    print(week_data)
    '''

    '''
    season_data = nfl.import_schedules([2022])
    season_data.to_csv("./data/season_data.csv")
    '''

    data = pd.read_csv("./data/nfl_elo.csv")
    elo_last = {}

    for _, row in data.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        elo1_post = row["elo1_post"]
        elo2_post = row["elo2_post"]
        qb1 = row["qb1"]
        qb2 = row["qb2"]

        elo_last[team1] = (qb1, elo1_post)
        elo_last[team2] = (qb2, elo2_post)
    # print the values in the dictionary in sorted order
    for key in sorted(elo_last):
        print("%s: %s" % (key, elo_last[key]))
    return

if __name__ == "__main__":
    main()