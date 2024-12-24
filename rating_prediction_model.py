#import libraries
import pandas as pd
import datetime
import numpy as np
import math
from IPython.display import clear_output
import unicodedata
from rapidfuzz import fuzz, process
from pathlib import Path


def split_up_totals(table):
    table_totals = table.iloc[-1]
    table = table.drop(index=table.index[-1])
    return table, table_totals


def get_cdf(value, dictionary):
    tot = 0
    s = 0
    for key in dictionary:
        tot += dictionary[key]
        if key <= value:
            s += dictionary[key]
    return float(s)/float(tot)


def get_weights(row):
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
    weight0 = min(1, row['GP0']*0.1)
    weight1 = 1 - weight0
    return weight0 * row['average_minutes_coeff_0'] + weight1 * row['average_minutes_coeff_1']

def get_overall_rating(row):
    if isinstance(row['weights'], np.ndarray):
        return row['weights']@np.array([row['rating0'], row['rating1'], row['rating2'], row['career_rating']])
    else:
        return 0

#get the start and end of training
start_training_year = int(input("Enter the year you want to start training: (year of season beginning)"))
end_training_year = int(input("Enter the season you want to stop training: (year of season beginning, and we'll test on the season after that; 5 year training set recommended; full data is available only if the training stops at 2022 or 2023)"))

# Define the base directory (relative to the current script)
base_dir = Path(__file__).parent

#initialize data (games, players, injuries tables, total_games, correct_predictions)
games = pd.read_csv(base_dir / "all_games.csv")
games['game_date'] = pd.to_datetime(games['game_date'])
train_games = games[(games['game_date'] >= datetime.datetime(start_training_year,10,1)) & (games['game_date'] <= datetime.datetime(end_training_year+1,9,30))]
test_games = games[(games['game_date'] >= datetime.datetime(end_training_year+1,10,1)) & (games['game_date'] <= datetime.datetime(end_training_year+2,9,30))]
test_games['will_home_win?'] = 'L'
first_test_game_date = test_games['game_date'].min()
last_test_game_date = test_games['game_date'].max()
players = pd.read_csv(base_dir / "all_players.csv")
players['To'] = players['To'].astype(int)
players = players[players['To'] >= 2003]
injuries = pd.read_csv(base_dir / "all_injuries.csv")
injuries['Date'] = pd.to_datetime(injuries['Date'])
injuries = injuries[(injuries['Date'] >= datetime.datetime(2003,10,28)) & (injuries['Date'] <= last_test_game_date)]
total_games = len(test_games)
correct_predictions = 0

#declare dates, injury mask
all_dates = pd.date_range(first_test_game_date, last_test_game_date).tolist()
injury_mask = {date: False for date in all_dates}

#add a bunch of columns to players
players['GP'] = 0
players['sum'] = 0
players['GP0'] = 0
players['sum0'] = 0
players['GP1'] = 0
players['sum1'] = 0
players['GP2'] = 0
players['sum2'] = 0
players['injured'] = False
players['sumusage'] = 0
players['sumusage0'] = 0
players['sumusage1'] = 0
players['career_rating'] = 0
players['career_avgusg'] = 0
players['rating0'] = 0
players['rating1'] = 0
players['rating2'] = 0
players['minutes_coeff_sum_1'] = 0
players['minutes_coeff_sum_0'] = 0
players['average_minutes_coeff_1'] = 0
players['average_minutes_coeff_0'] = 0

#stop_year is the year we stop our training set - process_game function
def process_game(row, b, usg, gmsc, rtg, stop_year, is_in_training):
    global players
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
    home_bs = pd.read_csv(base_dir / "data" / "Box_Scores" / (str(year) + str(month) + str(day) + home_team + away_team + "_2.csv"))
    #home_bs = pd.read_csv(r"C:\NBA\PythonProject\Box_Scores\\" + str(year) + str(month) + str(day) + home_team + away_team + "_2.csv")
    away_new = pd.read_csv(base_dir / "data" / "Box_Scores" / (str(year) + str(month) + str(day) + home_team + away_team + "_4.csv"))
    #away_new = pd.read_csv(r"C:\NBA\PythonProject\Box_Scores\\" + str(year) + str(month) + str(day) + home_team + away_team + "_4.csv")
    home_new = pd.read_csv(base_dir / "data" / "Box_Scores" / (str(year) + str(month) + str(day) + home_team + away_team + "_5.csv"))
    #home_new = pd.read_csv(r"C:\NBA\PythonProject\Box_Scores\\" + str(year) + str(month) + str(day) + home_team + away_team + "_5.csv")
    #clear_output(wait=True)
    #print(str(year) + str(month) + str(day) + home_team + away_team)
    home_bs, home_totals = split_up_totals(home_bs)
    total_team_mins = int(home_totals['MP'])
    home_new.apply(lambda row: change_rating(row, home_team, year, b, usg, gmsc, rtg, total_team_mins, stop_year, date, is_in_training), axis=1)
    away_new.apply(lambda row: change_rating(row, away_team, year, b, usg, gmsc, rtg, total_team_mins, stop_year, date, is_in_training), axis=1)

def change_rating(row, team, year, b, usg, gmsc, rtg, team_mins, stop_year, date, is_in_training):
    global players
    player_name = row['Starters']
    player_row_in_players_table = players[players['Player'] == player_name]
    player_row_in_players_table = player_row_in_players_table.reset_index(drop=True)
    player_row_in_players_table_temp = player_row_in_players_table
    if player_row_in_players_table.shape[0] > 1:
        for idx, doppelganger in player_row_in_players_table.iterrows():
            if year > doppelganger['To'] or year < doppelganger['From'] - 1:
                player_row_in_players_table_temp = player_row_in_players_table_temp.drop(idx)
                player_row_in_players_table_temp.reset_index(drop=True)
        player_row_in_players_table = player_row_in_players_table_temp
    if player_row_in_players_table.shape[0] > 1:
        for idx, doppelganger in player_row_in_players_table.iterrows():
            if doppelganger['Team'] == team:
                player_row_in_players_table = player_row_in_players_table.iloc[idx]
                break
    if isinstance(player_row_in_players_table, pd.DataFrame) and player_row_in_players_table.shape[0] > 1:
        player_row_in_players_table = player_row_in_players_table.reset_index(drop=True)
        player_row_in_players_table = player_row_in_players_table.iloc[0]
    if isinstance(player_row_in_players_table, pd.DataFrame):
        player_row_in_players_table = player_row_in_players_table.reset_index(drop=True)
        player_row_in_players_table = player_row_in_players_table.iloc[0]
    mask = (players[['Player', 'From']] == player_row_in_players_table[['Player', 'From']]).all(axis=1)
    if player_row_in_players_table['Team'] != team:
        players.loc[mask, 'Team'] = team
    min_played = row['MP']
    splitup = row['MP'].split(':')
    splitup[0] = int(splitup[0])
    splitup[1] = float(splitup[1])/60
    min_played = splitup[0] + splitup[1]
    minutes_coeff = (6.85*min_played)/team_mins
    if minutes_coeff > 1:
        minutes_coeff = 1
    usg_coeff = usg[float(row['USG%'])]
    gmsc_coeff = gmsc[float(row['GmSc'])]*20
    rtg_coeff = rtg[int(row['ORtg']) - int(row['DRtg'])]*20
    # print(str(K_factor) + " " + str(usg_coeff) + " " + str(minutes_coeff) + " " + str(a) + " " + str(gmsc_coeff) + " " + str(rtg_coeff) + " " + str(pm) + " " + str(t))
    score = usg_coeff*minutes_coeff*(b*gmsc_coeff + (1-b)*rtg_coeff)*5
    if score < -100:
        score = -100
    if score > 100:
        score = 100
    if player_name == "Tony Mitchell":
        if team == "MIL":
            mask = (players['Player'] == "Tony Mitchell") & (players['From'] == 2014) & (players['Ht'] == "6-6")
        else:
            mask = (players['Player'] == "Tony Mitchell") & (players['From'] == 2014) & (players['Ht'] == "6-8")
    players.loc[mask, 'GP'] = players.loc[mask, 'GP'].item() + 1
    players.loc[mask, 'sum'] = players.loc[mask, 'sum'].item() + score
    players.loc[mask, 'sumusage'] = players.loc[mask, 'sumusage'].item() + float(row['USG%'])
    if not(is_in_training):
        players.loc[mask, 'GP0'] = players.loc[mask, 'GP0'].item() + 1
        players.loc[mask, 'sum0'] = players.loc[mask, 'sum0'].item() + score
        players.loc[mask, 'sumusage0'] = players.loc[mask, 'sumusage0'].item() + float(row['USG%'])
        players.loc[mask, 'rating0'] =  float(players.loc[mask, 'sum0'].iloc[0])/float(players.loc[mask, 'GP0'].iloc[0])
        players.loc[mask, 'minutes_coeff_sum_0'] = players.loc[mask, 'minutes_coeff_sum_0'].item() + minutes_coeff
        players.loc[mask, 'average_minutes_coeff_0'] = float(players.loc[mask, 'minutes_coeff_sum_0'].iloc[0])/float(players.loc[mask, 'GP0'].iloc[0])
    else:
        k = stop_year - date.year
        month = date.month
        if date.month >= 10 and k <= 1:
            players.loc[mask, 'GP' + str(k+1)] = players.loc[mask, 'GP' + str(k+1)].item() + 1
            players.loc[mask, 'sum' + str(k+1)] = players.loc[mask, 'sum' + str(k+1)].item() + score
            players.loc[mask, 'rating' + str(k+1)] = float(players.loc[mask, 'sum' + str(k+1)].iloc[0])/float(players.loc[mask, 'GP' + str(k+1)].iloc[0])
            if k == 0:
                players.loc[mask, 'sumusage1'] = players.loc[mask, 'sumusage1'].item() + float(row['USG%'])
                players.loc[mask, 'minutes_coeff_sum_1'] = players.loc[mask, 'minutes_coeff_sum_1'].item() + minutes_coeff
                players.loc[mask, 'average_minutes_coeff_1'] = float(players.loc[mask, 'minutes_coeff_sum_1'].iloc[0])/float(players.loc[mask, 'GP1'].iloc[0])
        elif date.month <= 9 and k <= 0:
            players.loc[mask, 'GP' + str(k+2)] += 1
            players.loc[mask, 'sum' + str(k+2)] += score
            players.loc[mask, 'rating' + str(k+2)] = (float(players.loc[mask, 'sum' + str(k+2)].iloc[0]))/(float(players.loc[mask, 'GP' + str(k+2)].iloc[0]))
            if k == -1:
                players.loc[mask, 'sumusage1'] = players.loc[mask, 'sumusage1'].item() + float(row['USG%'])
                players.loc[mask, 'minutes_coeff_sum_1'] = players.loc[mask, 'minutes_coeff_sum_1'].item() + minutes_coeff
                players.loc[mask, 'average_minutes_coeff_1'] = float(players.loc[mask, 'minutes_coeff_sum_1'].iloc[0])/float(players.loc[mask, 'GP1'].iloc[0])
    players.loc[mask, 'career_rating'] = float(players.loc[mask, 'sum'].iloc[0])/float(players.loc[mask, 'GP'].iloc[0])
    players.loc[mask, 'career_avgusg'] = float(players.loc[mask, 'sumusage'].iloc[0])/float(players.loc[mask, 'GP'].iloc[0])

#injury functions
def injury_updates(injury_table, date):
    global players
    global injury_mask
    if injury_mask[date] == False:
        injuries_that_date = injury_table[injury_table['Date']==date]
        injuries_that_date.apply(lambda row: process_injury_row(row, date), axis=1)
        injury_mask[date] = True

def process_injury_row(row, date):
    global players
    #clear_output(wait=True)
    year = date.year
    name = row['Acquired']
    team = row['Team']
    player_returning = True
    if pd.isna(name):
        name = row['Relinquished']
        player_returning = False
    name_row_in_players = players[players['Player']==name]
    name_row_in_players = name_row_in_players.reset_index(drop=True)
    name_row_in_players_temp = name_row_in_players
    if name_row_in_players.shape[0] > 1:
        players_removed = 0
        for idx, doppelganger in name_row_in_players.iterrows():
            if  (doppelganger['injured'] != player_returning) or (year > doppelganger['To']) or (year < doppelganger['From'] - 1) or (doppelganger['Team'] != team):
                name_row_in_players_temp = name_row_in_players_temp.drop(idx-players_removed)
                name_row_in_players_temp.reset_index(drop=True)
                players_removed += 1
        name_row_in_players = name_row_in_players_temp
    if isinstance(name_row_in_players, pd.DataFrame):
        name_row_in_players = name_row_in_players.iloc[0]
    mask = (players[['Player', 'From']] == name_row_in_players[['Player', 'From']]).all(axis=1)
    if player_returning and date.month >= 10:
        players.loc[mask, 'To'] = year + 1
    if player_returning and date.month <= 9:
        players.loc[mask, 'To'] = year
    players.loc[mask, 'injured'] = (not player_returning)

def fuzzy_match(name, choices):
    global players
    match = process.extractOne(name, choices, scorer=fuzz.WRatio)
    return match[0]

def replace_team(row):
    global players
    #clear_output(wait=True)
    name = row['Name']
    #print(row['New Team'])
    name_match = fuzzy_match(name, players['Player'])
    players.loc[(players['Player']==name_match), 'Team'] = row['New Team']
    return row

def predict_winner_and_process(row, b, hca, usage_coeffs, gmsc_coeffs, rtg_coeffs, end_training_year, is_in_training):
    global players
    global injury_mask
    global injuries
    global correct_predictions
    #First we get injury updates and see who's playing
    date = row['game_date']
    injury_updates(injuries, date)
    home_team = row['matchup_home'][:3]
    away_team = row['matchup_home'][-3:]
    home_wl = -1
    if row['wl_home'] == 'W':
        home_wl = 1
    injury_updates(injuries, date)
    year = date.year
    month = date.month
    day = date.day
    #Then we identify the teams' players *who aren't injured*
    home_players = players[(players['Team']==home_team) & (players['injured']==False) & (players['To']>=year)]
    away_players = players[(players['Team']==away_team) & (players['injured']==False) & (players['To']>=year)]
    #Then we figure out how to assign weights of the previous seasons (both rating weights and usage weights) via their games played
    home_players['weights'] = home_players.apply(get_weights, axis=1)
    away_players['weights'] = away_players.apply(get_weights, axis=1)
    home_players['usage_wt_list'] = home_players.apply(get_usage_weights, axis=1)
    away_players['usage_wt_list'] = away_players.apply(get_usage_weights, axis=1)
    #Then we assign weights for the sum based on their usage rates
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
    #Then we compute the weighted average and predict the winner!
    home_players['weighted_rating'] = home_players.apply(get_overall_rating, axis=1)
    away_players['weighted_rating'] = away_players.apply(get_overall_rating, axis=1)
    home_ratings_array = np.array(home_players['weighted_rating'])
    away_ratings_array = np.array(away_players['weighted_rating'])
    home_rating = home_wt_array@home_ratings_array
    away_rating = away_wt_array@away_ratings_array
    if home_rating + hca >= away_rating:
        row['will_home_win?'] = 'W'
    else: row['will_home_win?'] = 'L'
    if row['wl_home'] == row['will_home_win?']:
        correct_predictions += 1
    #Finally, we process the games and get the new data before moving on to the next one...
    process_game(row, b, usage_coeffs, gmsc_coeffs, rtg_coeffs, end_training_year, is_in_training)
    return row


#get usage rate, game score, offrtg-defrtg histograms
end_year = end_training_year + 1
usage_histos_list = [pd.read_csv(base_dir / "data" / "Histograms" / ("usg_rts_histogram_" + str(year) + "-" + str(year + 1) + ".csv")) for year in list(range(start_training_year,end_year))]
#usage_histos_list = [pd.read_csv(r"C:\NBA\PythonProject\Histograms\usg_rts_histogram_" + str(year) + "-" + str(year + 1) + ".csv") for year in list(range(start_training_year,end_year))]
gmsc_histos_list = [pd.read_csv(base_dir / "data" / "Histograms" / ("gmscs_histogram_" + str(year) + "-" + str(year + 1) + ".csv")) for year in list(range(start_training_year,end_year))]
#gmsc_histos_list = [pd.read_csv(r"C:\NBA\PythonProject\Histograms\gmscs_histogram_" + str(year) + "-" + str(year + 1) + ".csv") for year in list(range(start_training_year,end_year))]
rtg_histos_list = [pd.read_csv(base_dir / "data" / "Histograms" / ("ortg-drtg_histogram_" + str(year) + "-" + str(year + 1) + ".csv")) for year in list(range(start_training_year,end_year))]
#rtg_histos_list = [pd.read_csv(r"C:\NBA\PythonProject\Histograms\ortg-drtg_histogram_" + str(year) + "-" + str(year + 1) + ".csv") for year in list(range(start_training_year,end_year))]
usage_histos_list[0].columns = ['score', 'frequency']
gmsc_histos_list[0].columns = ['score', 'frequency']
rtg_histos_list[0].columns = ['score', 'frequency']
usage_scores = usage_histos_list[0]['score']
gmsc_scores = gmsc_histos_list[0]['score']
rtg_scores = rtg_histos_list[0]['score']
usage_histo = {score: 0 for score in usage_scores}
gmsc_histo = {score: 0 for score in gmsc_scores}
rtg_histo = {score: 0 for score in rtg_scores}
for usg_histo_season, gmsc_histo_season, rtg_histo_season in zip(usage_histos_list, gmsc_histos_list, rtg_histos_list):
    usg_histo_season.columns = ['score', 'frequency']
    gmsc_histo_season.columns = ['score', 'frequency']
    rtg_histo_season.columns = ['score', 'frequency']
    for key in usage_histo:
        mask = (usg_histo_season['score']==key)
        usage_histo[key] += usg_histo_season.loc[mask, 'frequency']
    for key in gmsc_histo:
        mask = (gmsc_histo_season['score']==key)
        gmsc_histo[key] += gmsc_histo_season.loc[mask, 'frequency']
    for key in rtg_histo:
        mask = (rtg_histo_season['score']==key)
        rtg_histo[key] += rtg_histo_season.loc[mask, 'frequency']
for key in usage_histo:
    usage_histo[key] = int(usage_histo[key].iloc[0])
for key in gmsc_histo:
    gmsc_histo[key] = int(gmsc_histo[key].iloc[0])
for key in rtg_histo:
    rtg_histo[key] = int(rtg_histo[key].iloc[0])


#turn histograms into coefficient tables
gmsc_coeffs = {score : 0 for score in gmsc_scores}
for key in gmsc_coeffs:
    gmsc_coeffs[key] = 2*get_cdf(key, gmsc_histo)-1
odrtg_coeffs = {diff: 0 for diff in rtg_scores}
for key in odrtg_coeffs:
    odrtg_coeffs[key] = 2*get_cdf(key, rtg_histo)-1
usgrt_coeffs = {rate : 0 for rate in usage_scores}
for key in usgrt_coeffs:
    usgrt_coeffs[key] = get_cdf(key, usage_histo)

#predictions - run testing
#correct_predictions_list = []
b_value = 0.2
hca_value = 3
#i = 0
#run training
train_games.apply(lambda row: process_game(row, b_value, usgrt_coeffs, gmsc_coeffs, odrtg_coeffs, end_training_year, True), axis=1)
#deal with transcations between training and testing
#games = pd.read_csv(base_dir / "all_games.csv")
transactions = pd.read_csv(base_dir / "data" / ("transactions" + str(end_year) + ".csv"), encoding="windows-1252", header=0)
#transactions = pd.read_csv(r"C:\NBA\PythonProject\transactions" + str(end_year) + ".csv", encoding="windows-1252", header=0)
transactions.apply(replace_team, axis=1)
test_games = test_games.apply(lambda row: predict_winner_and_process(row, b_value, hca_value, usgrt_coeffs, gmsc_coeffs, odrtg_coeffs, end_training_year, False), axis=1)
test_games.to_csv(base_dir / ("bigtest_changed_paths" + "_" + str(b_value) + "_" + str(hca_value) + "_" + str(start_training_year) + " " + str(end_training_year) + ".csv"))
#test_games.to_csv(r"C:\NBA\PythonProject\bigtest_changed_paths" + "_" + str(b_value) + "_" + str(hca_value) + "_" + str(start_training_year) + " " + str(end_training_year) + ".csv")
players.to_csv(base_dir / ("players_bigtest_changed_paths" + "_" + "_" + str(b_value) + "_" + str(hca_value) + "_" + str(start_training_year) + " " + str(end_training_year) + ".csv"))
#players.to_csv(r"C:\NBA\PythonProject\players_bigtest_without_starting_year" + "_" + "_" + str(b_value) + "_" + str(hca_value) + "_" + str(start_training_year) + " " + str(end_training_year) + ".csv")
#correct_predictions_list.append(correct_predictions)
#correct_predictions_series = pd.Series(correct_predictions_list)
#correct_predictions_series.to_csv(r"C:\NBA\PythonProject\predictions_bigtest_starting_year" + "_" + str(b_value) + "_" + str(start_training_year) + " " + str(end_training_year) + ".csv")
#i += 1
print(str(correct_predictions)+"/"+str(total_games)+", accuracy:" + str(float(correct_predictions)/float(total_games)))