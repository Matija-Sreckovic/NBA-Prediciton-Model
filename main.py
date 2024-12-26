#import libraries
import pandas as pd
import datetime
import numpy as np
import math
import unicodedata
from rapidfuzz import fuzz, process
from pathlib import Path
import Utilities as util
import histograms as hist
import predicting_and_processing as pp


def main():
    #get the start and end of training
    start_training_year = int(input("Enter the year you want to start training: (year of season beginning, recommended 2018 resp. 2019)"))
    end_training_year = int(input("Enter the season you want to stop training: (year of season beginning, recommended 2022 resp. 2023)"))
    end_year = end_training_year + 1
    # Define the base directory (relative to the current script)
    base_dir = Path(__file__).parent
    #initialize data (games, players, injuries tables, total_games, correct_predictions)
    games = pd.read_csv(base_dir / "data" / "all_games.csv")
    games['game_date'] = pd.to_datetime(games['game_date'])
    train_games = games[(games['game_date'] >= datetime.datetime(start_training_year, 10, 1)) & (games['game_date'] <= datetime.datetime(end_training_year+1, 9, 30))]
    test_games = games[(games['game_date'] >= datetime.datetime(end_training_year+1, 10, 1)) & (games['game_date'] <= datetime.datetime(end_training_year+2, 9, 30))]
    test_games['will_home_win?'] = 'L'
    first_test_game_date = test_games['game_date'].min()
    last_test_game_date = test_games['game_date'].max()
    players = pd.read_csv(base_dir / ("trained_data_players_" + str(start_training_year) + "_" + str(end_training_year) + ".csv"))
    players['To'] = players['To'].astype(int)
    players = players[players['To'] >= 2003]
    injuries = pd.read_csv(base_dir / "data" / "all_injuries.csv")
    injuries['Date'] = pd.to_datetime(injuries['Date'])
    injuries = injuries[(injuries['Date'] >= datetime.datetime(2003, 10, 28)) & (injuries['Date'] <= last_test_game_date)]
    total_games = len(test_games)
    correct_predictions = 0
    #add a bunch of columns to players - only needed if training
    #players['GP'] = 0
    #players['sum'] = 0
    #players['GP0'] = 0
    #players['sum0'] = 0
    #players['GP1'] = 0
    #players['sum1'] = 0
    #players['GP2'] = 0
    #players['sum2'] = 0
    #players['injured'] = False
    #players['sumusage'] = 0
    #players['sumusage0'] = 0
    #players['sumusage1'] = 0
    #players['career_rating'] = 0
    #players['career_avgusg'] = 0
    #players['rating0'] = 0
    #players['rating1'] = 0
    #players['rating2'] = 0
    #players['minutes_coeff_sum_1'] = 0
    #players['minutes_coeff_sum_0'] = 0
    #players['average_minutes_coeff_1'] = 0
    #players['average_minutes_coeff_0'] = 0
    gmsc_coeffs, odrtg_coeffs, usgrt_coeffs = hist.get_histograms(start_training_year, end_year)
    #declare dates, injury mask
    all_dates = pd.date_range(first_test_game_date, last_test_game_date).tolist()
    injury_mask = {date: False for date in all_dates}
    #predictions - run testing
    #The b_value controls how much we take into account game score (20%) and how much we take into account OffRtg-DefRtg (80%) - this was tuned
    b_value = 0.2
    #hca = home court advantage
    hca_value = 3
    #run training - the data has already been trained, and since I don't want to upload a 200MB file to github, I just do the testing here :)
    #train_games.apply(lambda row: process_game(row, b_value, usgrt_coeffs, gmsc_coeffs, odrtg_coeffs, end_training_year, True), axis=1)
    #train_games.to_csv(base_dir / ("trained_data_games_" + str(start_training_year) + "_" + str(end_training_year) + ".csv"))
    #players.to_csv(base_dir / ("trained_data_players_" + str(start_training_year) + "_" + str(end_training_year) + ".csv"))
    #deal with transcations between training and testing
    transactions = pd.read_csv(base_dir / "data" / ("transactions" + str(end_year) + ".csv"), encoding="windows-1252", header=0)
    for idx, row in transactions.iterrows():
        players = pp.replace_team(row, players)
    for idx, row in test_games.iterrows():
        test_games.loc[idx, 'will_home_win?'], players, injury_mask, correct_predictions = pp.predict_winner_and_process(row, b_value, hca_value, usgrt_coeffs, gmsc_coeffs, odrtg_coeffs, end_training_year, False, players, injury_mask, injuries, correct_predictions)
    test_games.to_csv(base_dir / ("test_data_" + str(start_training_year) + "_" + str(end_training_year) + ".csv"))
    players.to_csv(base_dir / ("players_test_data_" + str(start_training_year) + "_" + str(end_training_year) + ".csv"))
    print(str(correct_predictions)+"/"+str(total_games)+", accuracy:" + str(float(correct_predictions)/float(total_games)))

if __name__ == "__main__":
    main()