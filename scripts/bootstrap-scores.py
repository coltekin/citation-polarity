#!/usr/bin/env python3
"""Bootstrap test for mean scores for each journal.
"""

import glob
import joblib
import time
import os
import argparse
import numpy as np
from collections import Counter

classes = ('neg', 'neu', 'pos')

def read_data(fname):
    aid, dt, scores = [], [], []
    with open(fname, "rt") as f:
        _ = next(f).strip().split('\t')
        for line in f:
            row = line.strip().split('\t')
            aid.append(row[0])
            dt.append(row[1])
            scores.append(np.array([float(row[x]) for x in range(2, 5)]))
    scores = np.array(scores)
    # softmax scores
    e = np.exp(scores - np.max(scores))
    prob = e / np.sum(e, axis=0)
    return aid, dt, scores, prob

ap = argparse.ArgumentParser()
ap.add_argument('input', nargs="+")
ap.add_argument('--iterations', '-i', type=int, default=10)
args = ap.parse_args()


print('Reading the data ', end="", flush=True)
st = time.time()
if os.path.exists('stats.joblib'):
    print('from stats.joblib...', end="", flush=True)
    data = joblib.load('stats.joblib')
else:
    print('from original files...', end="", flush=True)
    data = dict()
    for f in args.input:
        journal = f.split('_')[0]
        data[journal] = read_data(f)
    joblib.dump(data, 'stats.joblib')
print(f" done in {time.time() - st}s.")

sample = dict()
for journal in data:
    print(f"Sampling for {journal}...", end="", flush=True)
    st = time.time()
    sample[journal] = []
    d = data[journal][2]
    for i in range(args.iterations):
        ri = np.random.choice(d.shape[0], d.shape[0], replace=True)
        sample[journal].append(d[ri, :].mean(axis=0))
    sample[journal] = np.array(sample[journal])
    print(f" done in {time.time() - st}s.")

for journal in data:
    print(journal, end="")
    mean, se = np.mean(sample[journal], axis=0), np.std(sample[journal], axis=0)
    negpos = sample[journal][0, :] / sample[journal][2, :]
    print('\t'.join(["∓".join((str(m), str(s)))
                     for m, s in zip(mean, se)]), end="")
    print('\t'.join(("", str(negpos.mean()), str(negpos.std()))))
