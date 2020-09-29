"""
Implementation of Ramer-Douglas Peucker algorithm for use on numpy arrays and pandas Series/DataFrames.
"""

import numpy as np
import pandas as pd
from math import sqrt


def get_distance(start, end, p):
    d = np.linalg.norm(np.cross(end - start, start - p)) / np.linalg.norm(end - start)
    return d


def rdp(df, xcol, ycol, epsilon=0.001):
    col_0 = df.columns.get_loc(xcol)
    col_1 = df.columns.get_loc(ycol)
    np_data = df.to_numpy()
    index = 0
    d_max = 0

    start = np.array([np_data[0, col_0], np_data[0, col_1]])
    end = np.array([np_data[-1, col_0], np_data[-1, col_1]])

    for i, row in enumerate(np_data):
        d = get_distance(start, end, np.array([row[col_0], row[col_1]]))
        if d > d_max:
            index = i
            d_max = d

    if d_max > epsilon:
        result_0 = rdp(
            pd.DataFrame(np_data[: index + 1], columns=df.columns), xcol, ycol, epsilon
        )
        result_1 = rdp(
            pd.DataFrame(np_data[index:], columns=df.columns), xcol, ycol, epsilon
        )
        result = pd.concat(
            [result_0, result_1]
        )  # result_0 and result_1 are dataframes.
    else:
        result = df
    return result
