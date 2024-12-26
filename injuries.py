import pandas as pd


#injury functions
def injury_updates(injury_table, date, players_table, injury_mask):
    """
    Injuries explained: I have an injury table with a list of all injuries that ever happened, together with the player's name and the dates.
    There are 2 rows in the table for each injury: when it happened, and when the player recovered. I edited the injuries table so that
    all dates (both injury & recovery) are on dates when there were games, because I don't want to think about dates which are not mentioned in the
    games table.
    Since there are multiple games on most dates, I only want to check for injuries on the *first* occurrence of each date. So if injury_mask[date] == False,
    that means it's the *first* game of that date and that I *want* to update the injury statuses of players. (there is a Boolean column 'injured' in
    the players table). Once the injuries for that day have been processed, I set injury_mask[date] to True, and for the remaining games on that date,
    the program won't check for injuries.
    """
    if injury_mask[date] == False:
        injuries_that_date = injury_table[injury_table['Date']==date]
        for idx, row in injuries_that_date.iterrows():
            players_table = process_injury_row(row, date, players_table)
        injury_mask[date] = True
    return players_table, injury_mask


def remove_duplicates_injuries(name_row_in_players):
    name_row_in_players_temp = name_row_in_players
    if name_row_in_players.shape[0] > 1:
        players_removed = 0
        for idx, doppelganger in name_row_in_players.iterrows():
            if (doppelganger['injured'] != player_returning) or (year > doppelganger['To']) or (year < doppelganger['From'] - 1) or (doppelganger['Team'] != team):
                name_row_in_players_temp = name_row_in_players_temp.drop(idx-players_removed)
                name_row_in_players_temp.reset_index(drop=True)
                players_removed += 1
        name_row_in_players = name_row_in_players_temp
    if isinstance(name_row_in_players, pd.DataFrame):
        name_row_in_players = name_row_in_players.iloc[0]
    return name_row_in_players


def process_injury_row(row, date, players_table):
    """
    In the injuries table, there are 4 notable columns: 'Date', 'Team', 'Acquired', 'Relinquished'.
    If the player in question *recovered* on that date, his name is in the 'Acquired' column.
    If he was injured on that date, his name is in the 'Relinquished' columns. One of the columns is always empty.
    I update the players table, in particular the player in question's 'injured' column.
    """
    year = date.year
    #In each row of the injury table, a player has either recovered ('Acquired') or gotten injured ('Relinquished')
    name = row['Acquired']
    team = row['Team']
    player_returning = True
    if pd.isna(name):
        name = row['Relinquished']
        player_returning = False
    #Get the right player in the players table
    name_row_in_players = players_table[players_table['Player']==name]
    name_row_in_players = name_row_in_players.reset_index(drop=True)
    #Remove duplicates
    name_row_in_players = remove_duplicates_injuries(name_row_in_players)
    #Create a Boolean mask to change the players table
    mask = (players_table[['Player', 'From']] == name_row_in_players[['Player', 'From']]).all(axis=1)
    #If the player is returning from a long injury, make sure he's not listed as retired
    if player_returning and date.month >= 10:
        players_table.loc[mask, 'To'] = year + 1
    if player_returning and date.month <= 9:
        players_table.loc[mask, 'To'] = year
    #Set the player's injury status
    players_table.loc[mask, 'injured'] = (not player_returning)
    return players_table


