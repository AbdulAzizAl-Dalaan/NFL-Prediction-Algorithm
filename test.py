
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# create a dictionary with the player statistics
player_stats = {
    'Player 1': [2000, 20, 5, 1000, 10, 500, 5, 50, 10, 2, 3, 1, 80, 95, 40, 20],
    'Player 2': [3000, 25, 8, 500, 5, 1000, 8, 60, 12, 3, 2, 2, 75, 90, 45, 15],
    'Player 3': [1500, 10, 3, 800, 6, 700, 3, 70, 8, 1, 1, 0, 85, 80, 35, 25]
}

# create a pandas dataframe from the dictionary
df = pd.DataFrame.from_dict(player_stats, orient='index',
                             columns=['Pass Yards', 'Pass TDs', 'INTs', 'Rush Yards',
                                      'Rush TDs', 'Rec Yards', 'Rec TDs', 'Tackles',
                                      'Sacks', 'INTs', 'Forced Fumbles', 'Rec Fumbles',
                                      'FG%', 'XP%', 'Punt Avg', 'KO Return Avg'])

# output the dataframe to a csv file
df.to_csv('player_stats.csv', index_label='Player Name')

# read in the player stats from the csv file
df = pd.read_csv('player_stats.csv', index_col='Player Name')

# create the t-SNE object with perplexity of 10
tsne = TSNE(n_components=2, perplexity=2, random_state=42)

# fit the t-SNE model to the player stats
tsne_results = tsne.fit_transform(df)

# create a scatter plot of the t-SNE results
plt.scatter(tsne_results[:,0], tsne_results[:,1])

# add labels to the scatter plot
for i, player in enumerate(df.index):
    plt.annotate(player, (tsne_results[i,0], tsne_results[i,1]))

# display the scatter plot
plt.show()



