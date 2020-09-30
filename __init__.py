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


def rdp_reduce(np_data, col_0, col_1, epsilon=0.001):
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
        result_0 = rdp_reduce(np_data[: index + 1], col_0, col_1, epsilon)
        result_1 = rdp_reduce(np_data[index:], col_0, col_1, epsilon)
        result = np.concatenate([result_0, result_1])
    else:
        result = np.concatenate([[np_data[0]], [np_data[-1]]])
    return result


def rdp(df, xcol, ycol, epsilon=0.001):
    if isinstance(df, pd.DataFrame):
        col_0 = df.columns.get_loc(xcol)
        col_1 = df.columns.get_loc(ycol)
        np_data = df.to_numpy()
        result = rdp_reduce(np_data, col_0, col_1, epsilon=0.001)
        return pd.DataFrame(result, columns=df.columns).drop_duplicates(
            ignore_index=True
        )
    else:
        print("Not a dataframe")
        return
