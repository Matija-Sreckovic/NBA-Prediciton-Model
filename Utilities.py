#import libraries
import pandas as pd
import unicodedata
from rapidfuzz import fuzz, process


def split_up_totals(table):
    """
    Used to get the last row of a box score separately and remove it
    """
    table_totals = table.iloc[-1]
    table = table.drop(index=table.index[-1])
    return table, table_totals


def get_cdf(value, dictionary):
    """
    Gets the value of the empirical CDF based on a histogram in the form of a dictionary. The dict. keys are values of a discrete RV,
    while the values of the dict. are the number of occurrences of that value of the RV. Used to get usg_coeff, gmsc_coeff and rtg_coeff.
    """
    tot = 0
    s = 0
    for key in dictionary:
        tot += dictionary[key]
        if key <= value:
            s += dictionary[key]
    return float(s)/float(tot)


def fuzzy_match(name, choices):
    """
    Important for approximate matches - č/ć sometimes replaced by c, etc. Especially in the transactions and injuries tables.
    """
    match = process.extractOne(name, choices, scorer=fuzz.WRatio)
    return match[0]
