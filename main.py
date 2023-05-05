import nfl_data_py as nfl
import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import warnings
import matplotlib.pyplot as plt

columns_to_normalize = ['PF', 'PA', 'PD', 'Off Pts Rank', 'Off Yds Rank', 'Def Pts Rank', 'Def Yds Rank', 'T/G', 'SOS']
new_columns_ranks = ['PF Rank', 'PA Rank', 'PD Rank', 'Off-PTS Rank', 'Off-YDS Rank', 'Def-PTS Rank', 'Def-YDS Rank', 'T/G Rank', 'SoS Rank']

'''
Must analyze all data into rankings yearly between 1-32 all the following (x-axis):
    32 means best in the league, 1 means worst in the league

    - Points For (PF)
    - Points Against (PA)
    - Points Differential (PD)*
    - Off Points Rank (Off Pts Rank)*
    - Off Yds Rank (Off Yds Rank)
    - Def Points Against Rank (Def Pts Rank)
    - Def Yds Against Rank (Def Yds Rank)
    - Turnover Differential (T/G)*
    - Strength of Schedule (SOS)*

Then, make this a graph of these stats ranked in terms of Wins (y-axis):
    - Wins (W)

'''

decade = [year for year in range(2012, 2022)]
start_time = datetime.now()

def current_time():
    return (datetime.now() - start_time).total_seconds()

def get_fig1(nfl_data_frame):
    '''
    returns a figure of the first graph
    '''
    plt.figure(figsize=(16,10))
    for i, col in enumerate(new_columns_ranks):
        plt.subplot(3, 3, i+1)
        plt.scatter(nfl_data_frame[col], nfl_data_frame['Wins'])
        plt.xlabel(col)
        plt.ylabel('Wins')
        plt.title(f'Wins vs. {col}')

    plt.tight_layout()
    return plt

def get_fig2(nfl_data_frame):
    n_samples = nfl_data_frame.shape[0]

    # adjusting the perplexity value will change the number of records
    perplexity_val = min(n_samples - 1, 30)

    # Create a t-SNE model with 2 components
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val)

    # Fit and transform the t-SNE model on the normalized data
    tsne_results = tsne.fit_transform(nfl_data_frame[new_columns_ranks])

    # Add t-SNE results as new columns in the pandas DataFrame
    nfl_data_frame['t-SNE 1'] = tsne_results[:, 0]
    nfl_data_frame['t-SNE 2'] = tsne_results[:, 1]

    # Create a new column to store the highest ranked statistic for each row
    nfl_data_frame['Highest Stat'] = nfl_data_frame[new_columns_ranks].idxmax(axis=1)

    # Create the t-SNE scatterplot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=nfl_data_frame, x='t-SNE 1', y='Wins', hue='Highest Stat', palette='dark', size='Wins', legend="brief")
    plt.title('t-SNE Graph of Wins vs. t-SNE Dimension 1')
    return plt

def find_features(nfl_data_frame):
    '''
    Originally used a Decision Tree Classifier to find the best features to use, 
    but it was not as accurate as using the rankings of the stats, hence why I
    am using Random Forest Classifier to find the best features to use.
    '''

    # Create the features and target DataFrames
    x = nfl_data_frame[new_columns_ranks]
    y = nfl_data_frame['Wins']
    
    # Create a decision tree classifier model using scikit-learn
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the decision tree classifier model
    clf.fit(x, y)


    # ignore the warnings of filters without names of columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        selector = SelectFromModel(clf, prefit=True)
        x_selected = selector.transform(x)

    # Get the names of the filtered columns and print selected features
    selected_features = [feature for feature, selected in zip(new_columns_ranks, selector.get_support()) if selected]
    for feature in selected_features:
        print(feature)
    return



def get_decade_list(dts_df):
    '''
    returns a np array of dataframes for each year in the decade
    '''
    decade_list = []
    for year in decade:
        # get the year
        temp_df = dts_df.filter(pl.col("Year") == year)
        
        # get the teams that have a winning record
        year_df = temp_df.filter(pl.col("W") >= pl.col("L"))

        year_df = ( year_df.groupby("Tm").agg(
            [
                pl.col('W').sum().alias('Wins'),
                pl.col("L").sum().alias('Losses'),
                pl.col("PF").sum().alias('Points For'),
                pl.col("PA").sum().alias('Points Against'),
                pl.col("PD").sum().alias('Point Difference'),
                pl.col("Playoffs").first().alias("Playoffs"),
                pl.col("Off Pts Rank").sum().alias('Off Pts Rank'),
                pl.col("Off Yds Rank").sum().alias('Off Yds Rank'),
                pl.col("Def Pts Rank").sum().alias('Def Pts Rank'),
                pl.col("Def Yds Rank").sum().alias('Def Yds Rank'),
                pl.col("T/G").sum().alias('T/G'),
                pl.col("SoS").sum().alias('SoS')
            ]
            )
        )
        year_df = (
            year_df.with_columns(pl.col("Points For").rank(descending=False).alias("PF Rank"))
            .with_columns(pl.col("Points Against").rank(descending=True).alias("PA Rank"))
            .with_columns(pl.col("Point Difference").rank(descending=False).alias("PD Rank"))
            .with_columns(pl.col("Off Pts Rank").rank(descending=True).alias("Off-PTS Rank"))
            .with_columns(pl.col("Off Yds Rank").rank(descending=True).alias("Off-YDS Rank"))
            .with_columns(pl.col("Def Pts Rank").rank(descending=True).alias("Def-PTS Rank"))
            .with_columns(pl.col("Def Yds Rank").rank(descending=True).alias("Def-YDS Rank"))
            .with_columns(pl.col("T/G").rank(descending=True).alias("T/G Rank"))
            .with_columns(pl.col("SoS").rank(descending=True).alias("SoS Rank"))
        )
        year_df.drop_in_place("Tm")
        year_df.drop_in_place("Points For")
        year_df.drop_in_place("Points Against")
        year_df.drop_in_place("Point Difference")
        year_df.drop_in_place("Off Pts Rank")
        year_df.drop_in_place("Off Yds Rank")
        year_df.drop_in_place("Def Pts Rank")
        year_df.drop_in_place("Def Yds Rank")
        year_df.drop_in_place("T/G")
        year_df.drop_in_place("SoS")
        #print(year_df)
        #year_df.to_pandas() 
        decade_list.append(year_df)
    # convert to numpy array
    #decade_list = np.array(decade_list)
    return decade_list

def main():
    dts_df = pl.read_csv("./data/decade_team_stats.csv")
    elo_df = pl.read_csv("./data/nfl_elo.csv")

    #print(dts_df)
    decade_list = get_decade_list(dts_df)
    # for df in decade_list:
    #     print(df)

    # graph as TSNE
    full_decade_frame = pl.concat([year for year in decade_list])

    #sort by wins
    full_decade_frame = full_decade_frame.sort("Wins", descending=True)
    
    playoff_decade_frame = full_decade_frame.filter(pl.col("Playoffs").is_not_null())
    super_bowl_decade_frame = playoff_decade_frame.filter((pl.col("Playoffs") == "Won SB") | (pl.col("Playoffs") == "Lost SB"))    

    full_decade_frame = full_decade_frame.to_pandas()
    playoff_decade_frame = playoff_decade_frame.to_pandas()
    super_bowl_decade_frame = super_bowl_decade_frame.to_pandas()

    # get figures of Wins vs Stats
    get_fig1(full_decade_frame)
    get_fig1(playoff_decade_frame)
    get_fig1(super_bowl_decade_frame)

    # get figures of TSNE of Stats
    get_fig2(full_decade_frame)
    get_fig2(playoff_decade_frame)
    get_fig2(super_bowl_decade_frame)

    # Find top features from each Dataframe
    print("\nFinding Top Features for Winning Teams from 2012-2021")
    find_features(full_decade_frame)
    print("\nFinding Top Features for Playoff Teams from 2012-2021")
    find_features(playoff_decade_frame)
    print("\nFinding Top Features for Super Bowl Teams from 2012-2021")
    find_features(super_bowl_decade_frame)

    print(f"\nProgram finish in {current_time()} seconds")

    '''
    UNCOMMENT THE FOLLOWING TO SEE FEATURE AND TSNE PLOTS
    '''
    #plt.show()

    print("UNCOMMENT LINE 215 ABOVE THIS PRINT STATEMENT IN MAIN.PY TO SEE FEATURE AND TSNE PLOTS")
    return 0

if __name__ == "__main__":
    main()