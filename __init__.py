"""
Implementation of Ramer-Douglas Peucker algorithm for use on numpy arrays and pandas Series/DataFrames.
"""

import numpy as np
import pandas as pd
from math import sqrt


def get_distance(start, end, p):
    d = np.linalg.norm(np.cross(end - start, start - p)) / np.linalg.norm(end - start)
    return d


def rdp(df, columns, epsilon=0):
    index = 0
    d_max = 0
    col_0 = df.columns.get_loc(columns[0])
    col_1 = df.columns.get_loc(columns[1])

    start = np.array([df.to_numpy()[0, col_0], df.to_numpy()[0, col_1]])
    end = np.array([df.to_numpy()[0, col_0], df.to_numpy()[0, col_1]])

    for i, row in enumerate(df.to_numpy()):
        d = get_distance(start, end, np.array([row[col_0], row[col_1]]))
        if d > d_max:
            index = i
            d_max = d

