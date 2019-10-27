"""Some helper functions to create a time series, inject anomalies, shingle it and print it"""
import numpy as np
import matplotlib.pyplot as plt


def create_period(p_length):
    """Creates the period that will be copied throughout the timeseires"""
    period = np.zeros(p_length)
    for i in range(p_length):
        period[i] = np.sin((i * 2 * np.pi) / (p_length))
    return period


# n: total number of points in the time series
# p: length of the period. Period is a sin function
# sigma: std of added noise
def create_ts(n_points, p_length, sigma):
    """creates a time series by repeating a sin function with G(0,sigma) noise"""
    pts = np.zeros(n_points)
    pattern = create_period(p_length)
    for i in range(int(n_points / p_length)):
        j = p_length * i
        error = np.random.normal(0, sigma, p_length)
        pts[j:j + p_length] = pattern + error
    return pts


def shingle(series, dim):
    """takes a one dimensional series and shingles it into dim dimensions"""
    height = len(series) - dim + 1
    shingled = np.zeros((dim, height))
    for i in range(dim):
        shingled[i] = series[i:i + height]
    return shingled


def inject_anomalies(pts, l_anom, n_anom):
    """takes a time series and flattens l_anom values in n_anom random locations"""
    loc = np.random.randint(low=0, high=len(pts) - l_anom - 1, size=n_anom)
    for i in range(n_anom):
        for j in range(1, l_anom):
            pts[loc[i] + j] = pts[loc[i]]
    return pts, loc


def print_anomalies(pts, indices):
    """prints the time series and the anomalies found by partition"""
    plt.plot(indices, pts[indices], 'bo')
    plt.plot(pts, 'r')
    plt.show()
