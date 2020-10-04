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


def rdp_reduce(np_data, xcol_ind, ycol_ind, epsilon=0.001):
    index = 0
    d_max = 0
    start = [np_data[0, xcol_ind], np_data[0, ycol_ind]]
    end = [np_data[-1, xcol_ind], np_data[-1, ycol_ind]]

    for i, row in enumerate(np_data):
        d = get_distance(start, end, [row[xcol_ind], row[ycol_ind]])
        if d > d_max:
            index = i
            d_max = d

    if d_max > epsilon:
        result_0 = rdp_reduce(np_data[: index + 1], xcol_ind, ycol_ind, epsilon)
        result_1 = rdp_reduce(np_data[index:], xcol_ind, ycol_ind, epsilon)
        result = np.concatenate([result_0, result_1])
    else:
        result = np.concatenate([[np_data[0]], [np_data[-1]]])
    return result


def rdp(data, xcol, ycol, epsilon=0.001):
    if isinstance(data, pd.DataFrame):
        xcol_ind = data.columns.get_loc(xcol)
        ycol_ind = data.columns.get_loc(ycol)
        np_data = data.to_numpy()
        result = rdp_reduce(np_data, xcol_ind, ycol_ind, epsilon=0.001)
        return pd.DataFrame(result, columns=data.columns).drop_duplicates(
            ignore_index=True
        )
    elif isinstance(data, np.ndarray):
        result = rdp_reduce(data, xcol, ycol, epsilon=0.001)
        return pd.DataFrame(result).drop_duplicates(ignore_index=True).to_numpy()
    else:
        print("Not a dataframe or array")
        return
