import pandas as pd
import random

decade = [year for year in range(2012, 2021)]
HOME_ADVANTAGE = 48 # similar to FiveThirtyEight's home advantage
BYE_ADVANTAGE = 25  # similar to FiveThirtyEight's bye advantage
K_FACTOR = 20       # similar to FiveThirtyEight's k factor
SOS_FACTOR = 10     # Newely added based on important features from TSNE

def get_prob(elo1, elo2):
    '''
    Calculate the probability of team 1 winning
    '''
    return 1.0 / (10.0 ** (-(elo1 - elo2) / 400.0) + 1.0)

def update_elo(winner_elos, loser_elos):
    win_team_elo, win_team_qb_elo = winner_elos
    lose_team_elo, lose_team_qb_elo = loser_elos

    expected_win = get_prob(win_team_elo, lose_team_elo)
    expected_win_qb = get_prob(win_team_qb_elo, lose_team_qb_elo)

    win_team_elo += K_FACTOR * (1 - expected_win)
    lose_team_elo += K_FACTOR * (0 - (1 - expected_win))

    win_team_qb_elo += K_FACTOR * (1 - expected_win_qb)
    lose_team_qb_elo += K_FACTOR * (0 - (1 - expected_win_qb))

    return (win_team_elo, win_team_qb_elo), (lose_team_elo, lose_team_qb_elo)

def simulate_regular_season(elo_dict, schedule_dict, matchups_dict, team_records, sos_dict):
    '''
    Simulate the regular season
    '''
    with open('season_results.txt', 'w') as f:
        for week in range(1, 18):
            f.write(f'\nWeek {week}\n')
            for match in matchups_dict[week]:
                home_team = match[0]
                away_team = match[1]
                home_elo = elo_dict[home_team][0]
                away_elo = elo_dict[away_team][0]
                home_qb_elo = elo_dict[home_team][1]
                away_qb_elo = elo_dict[away_team][1]

                home_elo += HOME_ADVANTAGE + home_qb_elo
                away_elo += away_qb_elo

                home_elo += ((SOS_FACTOR * sos_dict[home_team]) * -1) # newly added
                away_elo += ((SOS_FACTOR * sos_dict[away_team]) * -1) # newly added

                if week > 5:
                    if home_team not in schedule_dict[week-1]:
                        home_elo += BYE_ADVANTAGE
                    if away_team not in schedule_dict[week-1]:
                        away_elo += BYE_ADVANTAGE

                if random.random() < get_prob(home_elo, away_elo):
                    winning_team, losing_team = home_team, away_team
                else:
                    winning_team, losing_team = away_team, home_team

                winner_team_elos, losing_team_elos = update_elo(elo_dict[winning_team], elo_dict[losing_team])

                elo_dict[winning_team] = winner_team_elos
                elo_dict[losing_team] = losing_team_elos 

                team_records[winning_team] = (team_records[winning_team][0] + 1, team_records[winning_team][1])
                team_records[losing_team] = (team_records[losing_team][0], team_records[losing_team][1] + 1)
                f.write(f"{winning_team:<3} def. {losing_team}\n")
    return

def get_playoff_teams(division_standing):
    '''
    Get the playoff teams
    '''
    afc_playoff_seed = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None}
    nfc_playoff_seed = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None}
    top_afc_teams = {}
    top_nfc_teams = {}
    for division in division_standing:
        # add the top 2 teams from each division
        if "NFC" in division:
            top_nfc_teams[division_standing[division][0][0]] = division_standing[division][0][1]
            del division_standing[division][0]
        else:
            top_afc_teams[division_standing[division][0][0]] = division_standing[division][0][1]
            del division_standing[division][0]

    afc_top4 = sorted(top_afc_teams.items(), key=lambda x: x[1][0], reverse=True)[:4]

    nfc_top4 = sorted(top_nfc_teams.items(), key=lambda x: x[1][0], reverse=True)[:4]

    for i in range(4):
        afc_playoff_seed[i+1] = afc_top4[i][0]
        nfc_playoff_seed[i+1] = nfc_top4[i][0]

    # get the next 3 seeds and sort by record
    afc_next = {}
    nfc_next = {}

    for division in division_standing:
        if "AFC" in division:
            afc_next[division_standing[division][0][0]] = division_standing[division][0][1]
            del division_standing[division][0]
        else:
            nfc_next[division_standing[division][0][0]] = division_standing[division][0][1]
            del division_standing[division][0]

    afc_bottom3 = sorted(afc_next.items(), key=lambda x: x[1][0], reverse=True)[:3]
    nfc_bottom3 = sorted(nfc_next.items(), key=lambda x: x[1][0], reverse=True)[:3]

    for i in range(3):
        afc_playoff_seed[i+5] = afc_bottom3[i][0]
        nfc_playoff_seed[i+5] = nfc_bottom3[i][0]

    return afc_playoff_seed, nfc_playoff_seed

def get_wildcard_matchups(playoff_seed):
    '''
    Get the playoff matchups
    '''
    wc_matchups = {}
    wc_bye = playoff_seed[1]
    wc_matchups[1] = (playoff_seed[2], playoff_seed[7])
    wc_matchups[2] = (playoff_seed[3], playoff_seed[6])
    wc_matchups[3] = (playoff_seed[4], playoff_seed[5])
    return wc_bye, wc_matchups

def simulate_conference_playoffs(wc_bye, wc_matchups, elo_dict, playoff_seed, division):
    '''
    Simulate the playoffs
    '''
    conference_champ = ''
    new_playoff_seed = playoff_seed.copy()
    with open("season_results.txt", "a") as f:
        f.write(f"\n{division} WILDCARD\n")
        for home_team, away_team in wc_matchups.values():
            home_elo = elo_dict[home_team][0]
            away_elo = elo_dict[away_team][0]
            home_qb_elo = elo_dict[home_team][1]
            away_qb_elo = elo_dict[away_team][1]

            home_elo += HOME_ADVANTAGE + home_qb_elo
            away_elo += away_qb_elo

            if random.random() < get_prob(home_elo, away_elo):
                winning_team, losing_team = home_team, away_team
            else:
                winning_team, losing_team = away_team, home_team

            winner_team_elos, losing_team_elos = update_elo(elo_dict[winning_team], elo_dict[losing_team])

            elo_dict[winning_team] = winner_team_elos
            elo_dict[losing_team] = losing_team_elos

            new_playoff_seed = {k: v for k, v in new_playoff_seed.items() if v != losing_team}
            f.write(f"{winning_team:<3} def. {losing_team}\n")

        new_playoff_seed_list = list(new_playoff_seed.values())

        # simulate the divisional round
        divisional_round_matchups = {}
        divisional_round_matchups[1] = (new_playoff_seed_list[0], new_playoff_seed_list[3])
        divisional_round_matchups[2] = (new_playoff_seed_list[1], new_playoff_seed_list[2])

        f.write(f"\n{division} DIVISIONAL ROUND\n")
        for home_team, away_team in divisional_round_matchups.values():
            home_elo = elo_dict[home_team][0]
            away_elo = elo_dict[away_team][0]
            home_qb_elo = elo_dict[home_team][1]
            away_qb_elo = elo_dict[away_team][1]

            home_elo += HOME_ADVANTAGE + home_qb_elo
            away_elo += away_qb_elo

            if random.random() < get_prob(home_elo, away_elo):
                winning_team, losing_team = home_team, away_team
            else:
                winning_team, losing_team = away_team, home_team

            winner_team_elos, losing_team_elos = update_elo(elo_dict[winning_team], elo_dict[losing_team])

            elo_dict[winning_team] = winner_team_elos
            elo_dict[losing_team] = losing_team_elos

            new_playoff_seed = {k: v for k, v in new_playoff_seed.items() if v != losing_team}
            f.write(f"{winning_team:<3} def. {losing_team}\n")
        
        # simulate the conference championship
        new_playoff_seed_list = list(new_playoff_seed.values())
        f.write(f"\n{division} CONFERENCE CHAMPIONSHIP\n")
        home_team = new_playoff_seed_list[0]
        away_team = new_playoff_seed_list[1]
        home_elo = elo_dict[home_team][0]
        away_elo = elo_dict[away_team][0]
        home_qb_elo = elo_dict[home_team][1]
        away_qb_elo = elo_dict[away_team][1]

        home_elo += HOME_ADVANTAGE + home_qb_elo
        away_elo += away_qb_elo

        if random.random() < get_prob(home_elo, away_elo):
            winning_team, losing_team = home_team, away_team
        else:
            winning_team, losing_team = away_team, home_team

        winner_team_elos, losing_team_elos = update_elo(elo_dict[winning_team], elo_dict[losing_team])

        elo_dict[winning_team] = winner_team_elos
        elo_dict[losing_team] = losing_team_elos

        f.write(f"{winning_team:<3} def. {losing_team}\n")

        conference_champ = winning_team
    return conference_champ
    

def simulate_superbowl(elo_dict, afc_team, nfc_team):
    '''
    Simulate the Super Bowl
    '''
    with open("season_results.txt", "a") as f:
        f.write("\nSUPERBOWL\n")
        home_team = afc_team
        away_team = nfc_team
        home_elo = elo_dict[home_team][0]
        away_elo = elo_dict[away_team][0]
        home_qb_elo = elo_dict[home_team][1]
        away_qb_elo = elo_dict[away_team][1]

        home_elo += HOME_ADVANTAGE + home_qb_elo
        away_elo += away_qb_elo

        if random.random() < get_prob(home_elo, away_elo):
            winning_team, losing_team = home_team, away_team
        else:
            winning_team, losing_team = away_team, home_team

        winner_team_elos, losing_team_elos = update_elo(elo_dict[winning_team], elo_dict[losing_team])

        elo_dict[winning_team] = winner_team_elos
        elo_dict[losing_team] = losing_team_elos

        f.write(f"{winning_team:<3} def. {losing_team}\n")

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

    division_standings = {
        "AFC East": {"BUF": (0, 0), "MIA": (0, 0), "NE": (0, 0), "NYJ": (0, 0)},
        "AFC North": {"BAL": (0, 0), "CIN": (0, 0), "CLE": (0, 0), "PIT": (0, 0)},
        "AFC South": {"HOU": (0, 0), "IND": (0, 0), "JAX": (0, 0), "TEN": (0, 0)},
        "AFC West": {"DEN": (0, 0), "KC": (0, 0), "LAC": (0, 0), "OAK": (0, 0)},
        "NFC East": {"DAL": (0, 0), "NYG": (0, 0), "PHI": (0, 0), "WAS": (0, 0)},
        "NFC North": {"CHI": (0, 0), "DET": (0, 0), "GB": (0, 0), "MIN": (0, 0)},
        "NFC South": {"ATL": (0, 0), "CAR": (0, 0), "NO": (0, 0), "TB": (0, 0)},
        "NFC West": {"ARI": (0, 0), "LAR": (0, 0), "SF": (0, 0), "SEA": (0, 0)}
    }


    team_records = {
        "BUF": (0, 0), "MIA": (0, 0), "NE": (0, 0), "NYJ": (0, 0),
        "BAL": (0, 0), "CIN": (0, 0), "CLE": (0, 0), "PIT": (0, 0),
        "HOU": (0, 0), "IND": (0, 0), "JAX": (0, 0), "TEN": (0, 0),
        "DEN": (0, 0), "KC": (0, 0), "LAC": (0, 0), "OAK": (0, 0),
        "DAL": (0, 0), "NYG": (0, 0), "PHI": (0, 0), "WAS": (0, 0),
        "CHI": (0, 0), "DET": (0, 0), "GB": (0, 0), "MIN": (0, 0),
        "ATL": (0, 0), "CAR": (0, 0), "NO": (0, 0), "TB": (0, 0),
        "ARI": (0, 0), "LAR": (0, 0), "SF": (0, 0), "SEA": (0, 0)
    }

    elo_data = pd.read_csv("./data/nfl_elo.csv")
    schdule_data = pd.read_csv("./data/season_data.csv")
    sos_data = pd.read_csv("./data/sos_data.csv")

    elo_last_team = {} # dictionary to hold the last elo rating for each team and qb

    for _, row in elo_data.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        elo1_post = row["elo1_post"]
        elo2_post = row["elo2_post"]
        qb1 = row["qb1"]
        qb2 = row["qb2"]
        qb1_post_elo = row["qbelo1_post"]
        qb2_post_elo = row["qbelo2_post"]

        elo_last_team[team1] = (elo1_post, qb1_post_elo)
        elo_last_team[team2] = (elo2_post, qb2_post_elo)

    schdule_dict = {}
    matchups_dict = {}
    for _, row in schdule_data.iterrows():
        week = row["week"]
        if week > 18:
            break
        team1 = row["home_team"]
        team2 = row["away_team"]

        # get the set of each team which plays that week
        if week not in schdule_dict:
            schdule_dict[week] = set()
            matchups_dict[week] = set()
        schdule_dict[week].add(team1)
        schdule_dict[week].add(team2)
        matchups_dict[week].add((team1, team2))
    

    sos_dict = {}
    for _, row in sos_data.iterrows():
        team = row["team"]
        sos = row["SoS"]
        sos_dict[team] = sos
    simulate_regular_season(elo_last_team, schdule_dict, matchups_dict, team_records, sos_dict)

    for division in division_standings:
        for team in division_standings[division]:
            division_standings[division][team] = team_records[team]

    # sort the division standings
    for division in division_standings:
        division_standings[division] = sorted(division_standings[division].items(), key=lambda x: x[1][0], reverse=True)
    
    afc_playoff_seed, nfc_playoff_seed = get_playoff_teams(division_standings)

    # get the wildcard matchups
    afc_bye, afc_wc_matchups = get_wildcard_matchups(afc_playoff_seed)
    nfc_bye, nfc_wc_matchups = get_wildcard_matchups(nfc_playoff_seed)

    # simulate the playoffs
    afc_champ = simulate_conference_playoffs(afc_bye, afc_wc_matchups, elo_last_team, afc_playoff_seed, "AFC")
    nfc_champ = simulate_conference_playoffs(nfc_bye, nfc_wc_matchups, elo_last_team, nfc_playoff_seed, "NFC")

    simulate_superbowl(elo_last_team, afc_champ, nfc_champ)

    print("FULL SEASON SIMULATION COMPLETED VIEW THE RESULTS IN THE season_results.txt FILE")

    return

if __name__ == "__main__":
    main()