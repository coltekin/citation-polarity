#!/usr/bin/env python3

import glob
import os
import numpy as np
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument('inputs', nargs="+")
args = ap.parse_args()

scores = dict()
for dirname in args.inputs:
    model = dirname.rsplit('-', 1)[0].split('-', 1)[-1]
    if model not in scores:
        scores[model] = []
    best_metric = 0
    best_logs = None
    for sfile in glob.glob( os.path.join(
        dirname, 'checkpoint-*', 'trainer_state.json')):
        with open(sfile, 'rt') as f:
            sdata = json.load(f)
        if sdata['best_metric'] > best_metric:
            best_metric = sdata['best_metric']
            for logs in sdata['log_history']:
                eval_f1 = logs.get('eval_f1')
                if eval_f1 and np.isclose(eval_f1, best_metric):
                    best_logs = logs
    scores[model].append(best_logs)

p, r, f = [], [], []
for model in scores:
    for sc, scname in zip((p, r, f), ('precision', 'recall', 'f1')):
        sclist = [x[f'eval_{scname}'] for x in scores[model]]
        sc.append((np.mean(sclist), np.std(sclist), model))

for i, (f1, f1std, model) in enumerate(sorted(f)):
    prec, precstd, _ = p[i]
    recall, recallstd, _ = r[i]
    e = '/'.join([str(x['epoch']) for x in scores[model]])
    print(f"{model:30} {100*f1:3.2f} ±{100*f1std:05.2f}", end=" ")
    print(f"{100*prec:3.2f} ±{100*precstd:05.2f}", end=" ")
    print(f"{100*recall:3.2f} ±{100*recallstd:05.2f}", end=" ")
    print(f"{e}")
