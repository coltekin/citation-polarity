#!/usr/bin/env python3
"""Bootstrap sampling for per-article data
"""
import glob
import joblib
import time
import os
import sys
import argparse
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

classes = ('neg', 'neu', 'pos')

def read_data(fname):
    with open(fname, "rt") as f:
        _ = next(f).strip().split('\t')
        scores = []
        for line in f:
            sc = [float(x) for x in line.strip().split('\t')]
#            rate = sc[0] / sc[2] if sc[2] != 0.0 else float('nan')
            rate = sc[0] / sc[2] if sc[2] != 0.0 else sc[0]
            sc.append(rate)
            scores.append(sc)
    scores = np.array(scores)
    return scores

ap = argparse.ArgumentParser()
ap.add_argument('input', nargs="+")
ap.add_argument('--iterations', '-i', type=int, default=10)
args = ap.parse_args()


print('Reading the data... ', end="", flush=True, file=sys.stderr)
data = dict()
st = time.time()
for f in args.input:
    journal = f.split('-')[1].replace('.txt', '')
    print(f" {journal}", end="", flush=True, file=sys.stderr)
    data[journal] = read_data(f)
print(f" done in {time.time() - st}s.", file=sys.stderr)

true_mean = dict()
true_median = dict()
sample_mean = dict()
sample_median = dict()
for journal in data:
    d = data[journal]
    true_mean[journal] = np.mean(d, axis=0)
    true_median[journal] = np.median(d, axis=0)
    print(f"Sampling for {journal}...", end="", flush=True, file=sys.stderr)
    st = time.time()
    sample_mean[journal] = []
    sample_median[journal] = []
    for i in range(args.iterations):
        ri = np.random.choice(d.shape[0], d.shape[0], replace=True)
        sample_mean[journal].append(d[ri, :].mean(axis=0))
        sample_median[journal].append(np.median(d[ri, :], axis=0))
    sample_mean[journal] = np.array(sample_mean[journal])
    sample_median[journal] = np.array(sample_median[journal])
    print(f" done in {time.time() - st}s.", file=sys.stderr)

head = ["journal"]
head += [f"mean_{cls}" for cls in ("neg", "neu", "pos", "n/p")]
head += [f"median_{cls}" for cls in ("neg", "neu", "pos", "n/p")]
head += [f"est_mean_{cls}" for cls in ("neg", "neu", "pos", "n/p")]
head += [f"est_median_{cls}" for cls in ("neg", "neu", "pos", "n/p")]
print("\t".join(head))
for journal in data:
    print(journal, end="\t")
    print("\t".join(str(x) for x in true_mean[journal]), end="\t")
    print("\t".join(str(x) for x in true_median[journal]), end="\t")
    est_mean = np.mean(sample_mean[journal], axis=0)
    se_mean  = np.std(sample_mean[journal], axis=0)
    est_median = np.mean(sample_median[journal], axis=0)
    se_median  = np.std(sample_median[journal], axis=0)
    print('\t'.join(["∓".join((str(m), str(s)))
                     for m, s in zip(est_mean, se_mean)]), end="\t")
    print('\t'.join(["∓".join((str(m), str(s)))
                     for m, s in zip(est_median, se_median)]))
