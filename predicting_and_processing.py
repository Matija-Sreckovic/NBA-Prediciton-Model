#import libraries
import pandas as pd
import datetime
import numpy as np
import math
import unicodedata
from rapidfuzz import fuzz, process
from pathlib import Path
import Utilities as util
import injuries as inj


def get_weights(row):
    """
    Used to calculate the rating of a player. 0 -> performance so far this season, 1 -> performance last season, 2 -> performance 2 seasons ago,
    no number -> career (rather, 5-year) performance.
    I chose the weights a bit arbitrarily and then tried to tune the most important ones; no improvements made.
    """
    gp0 = row['GP0']
    gp1 = row['GP1']
    gp2 = row['GP2']
    gp = row['GP']
    weights = -1
    gp0c = gp1c = gp2c = gpc = 0
    if gp > 0:
        gpc = 1
    if gp0 > 0:
        gp0c = 1
    if gp1 > 0:
        gp1c = 1
    if gp2 > 0:
        gp2c = 1
    match [gp0c, gp1c, gp2c, gpc]:
        case [0,0,0,0]:
            weights = 0
        case [0,0,0,1]:
            weights = [0,0,0,1]
        case [0,0,1,1]:
            weights = [0,0,0.5,0.5]
        case [0,1,0,1]:
            weights = [0,0.6,0,0.4]
        case [0,1,1,1]:
            weights = [0,0.4,0.3,0.3]
        case [1,0,0,1]:
            weights = [0.9,0,0,0.1]
        case [1,0,1,1]:
            weights = [0.65,0,0.1,0.25]
        case [1,1,0,1]:
            weights = [0.5,0.3,0,0.2]
        case [1,1,1,1]:
            weights = [0.5,0.2,0.1,0.2]
    if weights == -1:
        print(str(gp0) + " " + str(gp1) + " " + str(gp2) + " " + str(gp))
        raise ValueError("There's an error")
    if weights != 0:
        weights = np.array(weights)
    return weights


def get_usage_weights(row):
    """
    Used to get the "usage factor" of a player, i.e. to compute how much his rating should contribute to his team's rating when predicting the outcome
    of a game. Based on this season's performance, last season's performance and 5-yr performance.
    The triple is later @'ed with average usage rates from this season, last season and the 5-year career average.
    """
    gp0 = row['GP0']
    gp1 = row['GP1']
    gp = row['GP']
    gp0c = gp1c = gpc = 0
    weights = -1
    if gp0 > 0:
        gp0c = 1
    if 20 <= gp1 < 50:
        gp1c = 1
    elif gp1 >= 50:
        gp1c = 2
    if gp > 0:
        gpc = 1
    match [gp0c, gp1c, gpc]:
        case [0,0,0]:
            weights = 0
        case [0,0,1]:
            weights = [0,0,1]
        case [0,1,1]:
            weights = [0,0.3,0.7]
        case [0,2,1]:
            weights = [0,0.7,0.3]
        case [1,0,1]:
            weights = [0.7,0,0.3]
        case [1,1,1]:
            weights = [0.65,0.1,0.25]
        case [1,2,1]:
            weights = [0.6,0.2,0.2]
    if weights == -1:
        raise ValueError("There's an error")
    if weights != 0:
        weights = np.array(weights)
    return weights


def add_weighted_usage_column(row):
    """
    computes a player's weighted usage by dot-producting the list from the get_usage_weights fctn with the player's average usage rates
    from this season, last season and 5 seasons ago. Used in predict_winner_and_process
    """
    dot_product = 0
    if isinstance(row['usage_wt_list'], np.ndarray):
        el_0 = el_1 = 0
        if row['GP0'] > 0:
            el_0 = row['sumusage0']/row['GP0']
        if row['GP1'] > 0:
            el_1 = row['sumusage1']/row['GP1']
        el_2 = row['career_avgusg']
        dot_product = row['usage_wt_list']@np.array([el_0,el_1,el_2])
    return dot_product


def get_overall_min_coeff(row):
    """
    gets the minutes coefficient (used to compute the score in process_box_score) - returns dot product of weight vector ([1,0] after 10 games this season)
    and the average minutes coefficients for this and last season
    """
    weight0 = min(1, row['GP0']*0.1)
    weight1 = 1 - weight0
    return weight0 * row['average_minutes_coeff_0'] + weight1 * row['average_minutes_coeff_1']


def get_overall_rating(row):
    """
    gets the dot product of the 4 ratings and the weights chosen in get_weights
    """
    if isinstance(row['weights'], np.ndarray):
        return row['weights']@np.array([row['rating0'], row['rating1'], row['rating2'], row['career_rating']])
    else:
        return 0


#stop_year is the year we stop our training set - process_game function
def process_game(row, b, usg, gmsc, rtg, stop_year, is_in_training, players_table):
    """
    Important function that processes a row of the games dataframe. One row of 'games' contains basic info about a game. First we get the date and the
    teams that played, then we read the box scores for that game from the "Box_Scores" folder. We only use home_bs to get the team totals row,
    and we only need the team's total minutes played from that row (which is the same for both teams, hence we don't need the away box score).
    away_new and home_new are box score versions that only contain the info I need (minutes played, usage %, offRtg, defRtg, game score).
    Finally, in the end I call process_box_score twice - see below.
    """
    base_dir = Path(__file__).parent
    date = row['game_date']
    home_team = row['matchup_home'][:3]
    away_team = row['matchup_home'][-3:]
    year = date.year
    month = date.month
    day = date.day
    if month < 10:
        month = "0" + str(date.month)
    if day < 10:
        day = "0" + str(date.day)
    #Read the box scores.
    home_bs = pd.read_csv(base_dir / "data" / "Box_Scores" / (str(year) + str(month) + str(day) + home_team + away_team + "_2.csv"))
    away_new = pd.read_csv(base_dir / "data" / "Box_Scores" / (str(year) + str(month) + str(day) + home_team + away_team + "_4.csv"))
    home_new = pd.read_csv(base_dir / "data" / "Box_Scores" / (str(year) + str(month) + str(day) + home_team + away_team + "_5.csv"))
    print(str(year) + str(month) + str(day) + home_team + away_team)
    #The last row of home_bs is the team's totals (total min_played, rebounds, assists, points, etc). We only need MP from it, basically to see
    #if there was overtime or not.
    home_bs, home_totals = util.split_up_totals(home_bs)
    total_team_mins = int(home_totals['MP'])
    #Process the box scores and modify the players table accordingly
    players_table = process_box_score(home_team, date, b, home_new, players_table, total_team_mins, usg, gmsc, rtg, is_in_training, stop_year)
    players_table = process_box_score(away_team, date, b, away_new, players_table, total_team_mins, usg, gmsc, rtg, is_in_training, stop_year)
    return players_table
    #home_new.apply(lambda row: change_rating(row, home_team, year, b, usg, gmsc, rtg, total_team_mins, stop_year, date, is_in_training), axis=1)
    #away_new.apply(lambda row: change_rating(row, away_team, year, b, usg, gmsc, rtg, total_team_mins, stop_year, date, is_in_training), axis=1)


def add_coeffs_and_score_to_box_score(box_score, team_mins, usg, gmsc, rtg, b):
    box_score[['minutes', 'seconds']] = box_score['MP'].str.split(':', expand=True)
    box_score['min_played'] = box_score['minutes'].astype(float) + (box_score['seconds'].astype(float)/60)
    box_score['minutes_coeff'] = np.minimum(1, (6.85*box_score['min_played']/team_mins))
    box_score['usg_coeff'] = box_score['USG%'].astype(float).map(usg)
    box_score['gmsc_coeff'] = box_score['GmSc'].astype(float).map(gmsc)*20
    box_score['rtg_coeff'] = (box_score['ORtg'].astype(int) - box_score['DRtg'].astype(int)).map(rtg)*20
    box_score['score'] = box_score['usg_coeff']*box_score['minutes_coeff']*(b*box_score['gmsc_coeff'] + (1-b)*box_score['rtg_coeff'])*5
    box_score['score'] = np.maximum(-100, box_score['score'])
    box_score['score'] = np.minimum(100, box_score['score'])
    return box_score


def modify_players_to_process_table(players_to_process, is_in_training, stop_year, year):
    #'GP' = career games played, 'sum' = sum of all scores in the last 6 years, 'sumusage' = sum of all usage rates in the last 6 years
    # Suffix 0 = only this season, suffix 1 = last season, suffix 2 = 2 seasons ago.
    players_to_process['GP'] += 1
    players_to_process['sum'] += players_to_process['score']
    players_to_process['sumusage'] += players_to_process['USG%']
    if not(is_in_training):
        players_to_process['GP0'] += 1
        players_to_process['sum0'] += players_to_process['score']
        players_to_process['sumusage0'] += players_to_process['USG%']
        players_to_process['rating0'] =  players_to_process['sum0'].astype(float)/players_to_process['GP0'].astype(float)
        players_to_process['minutes_coeff_sum_0'] += players_to_process['minutes_coeff']
        players_to_process['average_minutes_coeff_0'] = players_to_process['minutes_coeff_sum_0'].astype(float)/players_to_process['GP0'].astype(float)
    else:
        k = stop_year - year
        #The season starts in October and ends in June (but in some exceptional cases it lasted until September).
        if date.month >= 10 and k <= 1:
            players_to_process['GP' + str(k+1)] += 1
            players_to_process['sum' + str(k+1)] += players_to_process['score']
            players_to_process['rating' + str(k+1)] = players_to_process['sum' + str(k+1)].astype(float)/players_to_process['GP' + str(k+1)].astype(float)
            if k == 0:
                players_to_process['sumusage1'] += players_to_process['USG%'].astype(float)
                players_to_process['minutes_coeff_sum_1'] += players_to_process['minutes_coeff']
                players_to_process['average_minutes_coeff_1'] = players_to_process['minutes_coeff_sum_1'].astype(float)/players_to_process['GP1'].astype(float)
        elif date.month <= 9 and k <= 0:
            players_to_process['GP' + str(k+2)] += 1
            players_to_process['sum' + str(k+2)] += players_to_process['score']
            players_to_process['rating' + str(k+2)] = (players_to_process['sum' + str(k+2)].astype(float))/(players_to_process['GP' + str(k+2)].astype(float))
            if k == -1:
                players_to_process['sumusage1'] += players_to_process['USG%'].astype(float)
                players_to_process['minutes_coeff_sum_1'] += players_to_process['minutes_coeff']
                players_to_process['average_minutes_coeff_1'] = players_to_process['minutes_coeff_sum_1'].astype(float)/players_to_process['GP1'].astype(float)
    players_to_process['career_rating'] = players_to_process['sum'].astype(float)/players_to_process['GP'].astype(float)
    players_to_process['career_avgusg'] = players_to_process['sumusage'].astype(float)/players_to_process['GP'].astype(float)
    # Drop box_score columns from the DataFrame
    players_to_process = players_to_process.drop(columns=['GmSc', 'MP',
       'USG%', 'ORtg', 'DRtg', 'minutes', 'seconds', 'min_played',
       'minutes_coeff', 'usg_coeff', 'gmsc_coeff', 'rtg_coeff', 'score'])
    return players_to_process


def process_box_score(team, date, b, box_score, players_table, team_mins, usg, gmsc, rtg, is_in_training, stop_year):
    """
    This one's a bit annoying. First we find the players that played in the big players table and remove duplicates. Result: players_to_process DF
    Then we add some columns to box_score which contain the values we actually need to compute the rating ('minutes_coeff', 'usg_coeff', 'gmsc_coeff',
    'rtg_coeff', 'score').
    We drop a useless column 'Unnamed: 0' which is present in both box_score and players_to_process and creates confusion when merging.
    Then we merge the two dataframes along player names.
    Then we update the data (GP = games played, sum = sum of scores, GP0 = games played this season, etc.).
    If is_in_training, that means it's not the *current* season, so we'd update e.g. GP1,GP2,GP but not GP0.
    If not is_in_training, then it's the *current* season, so we update GP0, rating0 etc.
    Finally, we drop the box score columns from players_to_process, we drop the rows of players_to_process from players, and we add back the modified
    versions of the rows.
    """
    year = date.year
    month = date.month
    #Find players, remove duplicates
    players_to_process = players_table[players_table['Player'].isin(box_score['Starters'])]
    players_to_process = players_to_process[~(players_to_process.duplicated(subset='Player', keep=False) & ((year > players_to_process['To']) | (year < players_to_process['From'] - 1)))]
    players_to_process = players_to_process.drop_duplicates(subset='Player', keep='first')
    players_to_process['Team'] = team
    #Add necessary columns to box_score that help compute the rating
    box_score = add_coeffs_and_score_to_box_score(box_score, team_mins, usg, gmsc, rtg, b)
    #Prepare for merging by renaming column, drop useless column 'Unnamed: 0', merge
    box_score.rename(columns={'Starters': 'Player'}, inplace=True)
    players_to_process = players_to_process.drop(columns=['Unnamed: 0'], errors='ignore')
    box_score = box_score.drop(columns=['Unnamed: 0'], errors='ignore')
    players_to_process = pd.merge(players_to_process, box_score, on='Player', how='inner')
    #modify the values of the players_to_process_table by modifying the players' ratings and other auxiliary parameters
    players_to_process = modify_players_to_process_table(players_to_process, is_in_training, stop_year, year)
    # Initialize an empty list to hold the indices to drop
    indices_to_replace = []
    # Iterate through each row of player_rows_to_process
    for idx, row in players_to_process.iterrows():
        # Find the indices in players that match both 'Player' and 'From' values from the current row
        matching_indices = players_table[(players_table['Player'] == row['Player']) & (players_table['From'] == row['From'])].index
        # Append the matching indices to the list
        indices_to_replace.extend(matching_indices)
    # Drop the original rows from players
    players_table = players_table.drop(indices_to_replace)
    players_table = pd.concat([players_table, players_to_process], ignore_index=True)
    return players_table


def replace_team(row, players_table):
    """
    Go through transactions row by row and change the team of the corresponding row of the players table.
    """
    name = row['Name']
    print(row['New Team'])
    name_match = util.fuzzy_match(name, players_table['Player'])
    players_table.loc[(players_table['Player']==name_match), 'Team'] = row['New Team']
    return players_table


def compute_ratings(home_players, away_players):
    #We figure out how to assign weights of the previous seasons (both rating weights and usage weights) via their games played
    home_players['weights'] = home_players.apply(get_weights, axis=1)
    away_players['weights'] = away_players.apply(get_weights, axis=1)
    home_players['usage_wt_list'] = home_players.apply(get_usage_weights, axis=1)
    away_players['usage_wt_list'] = away_players.apply(get_usage_weights, axis=1)
    #We assign weights for the sum based on their usage rates
    home_players['usage_wt'] = home_players.apply(add_weighted_usage_column, axis=1)
    away_players['usage_wt'] = away_players.apply(add_weighted_usage_column, axis=1)
    home_players['minutes_coeff'] = home_players.apply(get_overall_min_coeff, axis=1)
    away_players['minutes_coeff'] = away_players.apply(get_overall_min_coeff, axis=1)
    home_players['multiplied_wt'] = home_players['usage_wt']*home_players['minutes_coeff']
    away_players['multiplied_wt'] = away_players['usage_wt']*away_players['minutes_coeff']
    home_wt_array = np.array(home_players['multiplied_wt'])
    away_wt_array = np.array(away_players['multiplied_wt'])
    home_wt_array = home_wt_array/np.linalg.norm(home_wt_array)
    away_wt_array = away_wt_array/np.linalg.norm(away_wt_array)
    #We compute the weighted average and predict the winner!
    home_players['weighted_rating'] = home_players.apply(get_overall_rating, axis=1)
    away_players['weighted_rating'] = away_players.apply(get_overall_rating, axis=1)
    home_ratings_array = np.array(home_players['weighted_rating'])
    away_ratings_array = np.array(away_players['weighted_rating'])
    home_rating = home_wt_array@home_ratings_array
    away_rating = away_wt_array@away_ratings_array
    return home_rating, away_rating


def predict_winner_and_process(row, b, hca, usage_coeffs, gmsc_coeffs, rtg_coeffs, end_training_year, is_in_training, players_table, injury_mask, injuries, correct_predictions):
    """
    We get the appropriate weights for how much a player's score (itself weighted by his performance in the prev. seasons) should impact the team's score.
    These are based on a player's minutes played and usage rate.
    Then we compute the weighted average and predict the winner (the home team gets a 3-point bonus; this value was tuned)
    After predicting, we process the game and update the players table.
    """
    #First we get injury updates and see who's playing
    date = row['game_date']
    players_table, injury_mask = inj.injury_updates(injuries, date, players_table, injury_mask)
    home_team = row['matchup_home'][:3]
    away_team = row['matchup_home'][-3:]
    home_wl = -1
    if row['wl_home'] == 'W':
        home_wl = 1
    year = date.year
    month = date.month
    day = date.day
    #Then we identify the teams' players *who aren't injured* (or retired)
    home_players = players_table[(players_table['Team']==home_team) & (players_table['injured']==False) & (players_table['To']>=year)].copy()
    away_players = players_table[(players_table['Team']==away_team) & (players_table['injured']==False) & (players_table['To']>=year)].copy()
    #We compute their scores
    home_rating, away_rating = compute_ratings(home_players, away_players)
    if home_rating + hca >= away_rating:
        will_home_win = 'W'
    else: will_home_win = 'L'
    if row['wl_home'] == will_home_win:
        correct_predictions += 1
    #Finally, we process the game and get the new data before moving on to the next one...
    players_table = process_game(row, b, usage_coeffs, gmsc_coeffs, rtg_coeffs, end_training_year, is_in_training, players_table)
    return will_home_win, players_table, injury_mask, correct_predictions


