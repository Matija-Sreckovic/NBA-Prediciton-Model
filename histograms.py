import pandas as pd
from pathlib import Path
import Utilities as util

def get_histograms(start_training_year, end_year):
    #get usage rate, game score, offrtg-defrtg histograms. I have them stored separately for each season - they are then summed over all training seasons.
    #There are 3 of them - usage rating, game score and (Offensive Rating minus Defensive Rating). They help determine how big a player's usage rate/gamescore/offdefrtg was,
    #compared to all other performances in the training set. That way, I don't have to judge what constitutes a good/bad performance, the data does it for me.
    base_dir = Path(__file__).parent
    usage_histos_list = [pd.read_csv(base_dir / "data" / "Histograms" / ("usg_rts_histogram_" + str(year) + "-" + str(year + 1) + ".csv")) for year in list(range(start_training_year,end_year))]
    gmsc_histos_list = [pd.read_csv(base_dir / "data" / "Histograms" / ("gmscs_histogram_" + str(year) + "-" + str(year + 1) + ".csv")) for year in list(range(start_training_year,end_year))]
    rtg_histos_list = [pd.read_csv(base_dir / "data" / "Histograms" / ("ortg-drtg_histogram_" + str(year) + "-" + str(year + 1) + ".csv")) for year in list(range(start_training_year,end_year))]
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
            mask = (usg_histo_season['score'] == key)
            usage_histo[key] += usg_histo_season.loc[mask, 'frequency']
        for key in gmsc_histo:
            mask = (gmsc_histo_season['score'] == key)
            gmsc_histo[key] += gmsc_histo_season.loc[mask, 'frequency']
        for key in rtg_histo:
            mask = (rtg_histo_season['score'] == key)
            rtg_histo[key] += rtg_histo_season.loc[mask, 'frequency']
    for key in usage_histo:
        usage_histo[key] = int(usage_histo[key].iloc[0])
    for key in gmsc_histo:
        gmsc_histo[key] = int(gmsc_histo[key].iloc[0])
    for key in rtg_histo:
        rtg_histo[key] = int(rtg_histo[key].iloc[0])

    #turn histograms into coefficient tables
    gmsc_coeffs = {score: 0 for score in gmsc_scores}
    for key in gmsc_coeffs:
        gmsc_coeffs[key] = 2*util.get_cdf(key, gmsc_histo)-1
    odrtg_coeffs = {diff: 0 for diff in rtg_scores}
    for key in odrtg_coeffs:
        odrtg_coeffs[key] = 2*util.get_cdf(key, rtg_histo)-1
    usgrt_coeffs = {rate: 0 for rate in usage_scores}
    for key in usgrt_coeffs:
        usgrt_coeffs[key] = util.get_cdf(key, usage_histo)

    return gmsc_coeffs, odrtg_coeffs, usgrt_coeffs

