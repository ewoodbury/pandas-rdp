"""
Implementation of Ramer-Douglas Peucker algorithm for use on numpy arrays and pandas Series/DataFrames.
"""

import numpy as np
import pandas as pd
from math import sqrt


def get_distance(start, end, p):
    x_diff = end[0] - start[0]
    y_diff = end[1] - start[1]
    return abs(
        y_diff * p[0] - x_diff * p[1] + end[0] * start[1] - end[1] * start[0]
    ) / sqrt(y_diff ** 2 + x_diff ** 2)


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
        result = pd.concat([result_0, result_1])
    else:
        result = pd.DataFrame(
            np.concatenate([[np_data[0]], [np_data[-1]]]), columns=df.columns
        )
    return result.drop_duplicates(ignore_index=True)
