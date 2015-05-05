#!/bin/pyton3
# ref: http://sucrose.hatenablog.com/entry/2013/11/10/001555
import numpy as np
from pylab import *
import pandas
import sys

def kullback_leibler_divergence(p, q):
    return np.sum(np.where((p != 0) & (q != 0), p * np.log2(p / q), 0))

def jensen_shannon_divergence(p, q):
    m = (p + q) / 2.0
    return (kullback_leibler_divergence(p, m) + kullback_leibler_divergence(q, m)) / 2.0

def prepare_data_prob_file(csv_filepath):
    data_frame = pandas.read_csv(csv_filepath, delim_whitespace=True, dtype="float")
    X = data_frame.as_matrix()
    return X.T # transpose is not required

def calcurate_prob_distance(data_prob, distance_function):
    result = [distance_function(data_prob[0], data_prob[i]) for i in range(1, len(data_prob))]
    bar(range(len(result)), result)
    show()

def main():
    if len(sys.argv) != 2:
        print('Usage: # python3 %s filename' % sys.argv[0])
        quit()
    data_prob = prepare_data_prob_file(sys.argv[1])
    calcurate_prob_distance(data_prob, jensen_shannon_divergence)

if __name__ == '__main__':
    main()
