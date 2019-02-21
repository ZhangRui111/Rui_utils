"""
Copy from Zongpu Zhang's code.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv_manually(fname):
    with open(fname) as csv_file:
        lines = csv_file.readlines()
    lines = [i.strip().split(',') for i in lines]
    data = []
    for row in lines:
        data += row
    return data


def average_from_start(data):
    """Calculate the average from start. AVG([ 0 ~ i ])"""
    accum = 0
    avg = []
    for idx in range(len(data)):
        accum += data[idx]
        avg.append(accum / (idx + 1 + 1e-7))
    return avg


def running_average(data, interval=50):
    """Calculate the running average of given data. AVG([ i ~ i+interval ])"""
    avg = []
    for i in range(len(data) - interval):
        left = i
        right = min(i + interval, len(data))
        avg.append(sum(data[left:right]) / (right - left))
    return avg


def convolve_line(data, interval=200):
    """Convolve on an array of data, with interval specified as a parameter"""
    return np.convolve(
        data,
        np.ones((interval,)) / interval, mode='same'
    )[:-1 * interval]


def smoothen_line(data, N=300):
    """Smoothen a line by interpolation."""
    from scipy.interpolate import make_interp_spline, BSpline
    xold = range(len(data))
    xnew = np.linspace(xold[0], xold[-1], N)  # 300 represents number of points to make between T.min and T.max
    spl = make_interp_spline(xold, data, k=3)  # BSpline object
    power_smooth = spl(xnew)
    return xnew, power_smooth


all_lines = []
all_smooth_lines = []
all_titles = []
all_axs = []
all_configs = []
for rd in ['0', '1', '10', '20', '100']:
    # load csv file
    fname = '../logs/smooth_plot/{0}/rewards.csv'.format(rd)
    data = load_csv_manually(fname)
    # only set 1 if raw data is 1, otherwise 0
    data = [1 if float(i) == 1 else 0 for i in data]
    # average score rate from the beginning to the current point
    # avg_from_start = average_from_start(data)
    # running average
    running_avg = running_average(data, interval=50)
    smoothen_running_avg = smoothen_line(convolve_line(running_avg, interval=200))
    # save
    all_smooth_lines.append(smoothen_running_avg)
    all_configs.append({'alpha': 0.2})
    all_lines.append(running_avg[:-1 * 200])  # strip the tail due to smoothen
    all_titles.append(r'$\delta$={}'.format(rd))

# draw lines
for idx in range(len(all_lines)):
    p = plt.plot(all_lines[idx], alpha=all_configs[idx]['alpha'])
    color = p[-1].get_color()
    # Add smooth line
    p1 = plt.plot(all_smooth_lines[idx][0], all_smooth_lines[idx][1],
                  color=color, label=all_titles[idx])
    all_axs.append(p)
    all_axs.append(p1)

plt.xlabel('epochs')
plt.ylabel('Goal rate (running average in 50 epochs)')
plt.legend(loc='best')
plt.show()
