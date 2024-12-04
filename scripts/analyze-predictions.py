#!/usr/bin/env python3
"""Generate some tables from predictions produced by predict.py.
"""

import glob
import joblib
import argparse
import os
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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
args = ap.parse_args()


if os.path.exists('stats.joblib'):
    data = joblib.load('stats.joblib')
else:
    data = dict()
    for f in args.input:
        journal = os.path.basename(f).split('_')[0]
        data[journal] = read_data(f)
    joblib.dump(data, 'stats.joblib')

for j in data: #('frbot',): #data:
    perart = dict()
    monthly = dict()
    yearly = dict()
    prev_a = None
    for i, dt in enumerate(data[j][1]):
        m = dt.rsplit('-', 1)[0]
        y = dt.split('-', 1)[0]
        aid = data[j][0][i]
        if aid not in perart:
            perart[aid] = {'date': dt,'polarity': [], 'entropy': []}
        if m not in monthly:
            monthly[m] = {'articles': 0, 'polarity': [], 'entropy': []}
        if y not in yearly:
            yearly[y] = {'articles': 0, 'polarity': [], 'entropy': []}
        c = classes[np.argmax(data[j][2][i])]
        e = -np.sum(data[j][3][i] * np.log2(data[j][3][i]))
        monthly[m]['polarity'].append(c)
        monthly[m]['entropy'].append(e)
        yearly[y]['polarity'].append(c)
        yearly[y]['entropy'].append(e)
        perart[aid]['polarity'].append(c)
        perart[aid]['entropy'].append(e)
        if prev_a != aid:
            monthly[m]['articles'] += 1
            yearly[y]['articles'] += 1
            if prev_a:
                c = Counter(perart[prev_a]['polarity'])
                perart[prev_a].update({
                    'pos': c['pos'],
                    'neg': c['neg'],
                    'neu': c['neu']
                })
            prev_a = aid
    if prev_a:
        c = Counter(perart[prev_a]['polarity'])
        perart[prev_a].update({
            'pos': c['pos'],
            'neg': c['neg'],
            'neu': c['neu']
        })

    with open(f"{j}-monthly.tsv", "wt") as fout:
        print('year', 'month', 'articles', 'neg', 'neu', 'pos',
              'entropy', 'entropy_sd', sep='\t', file=fout)
        for m in sorted(monthly):
            mdata = monthly[m]
            c = Counter(mdata['polarity'])
            mm, yy = m.split('-')
            print(mm, yy, mdata['articles'], c['neg'], c['neu'], c['pos'],
                  np.mean(mdata['entropy']), np.std(mdata['entropy']),
                  sep='\t', file=fout)
    with open(f"{j}-yearly.tsv", "wt") as fout:
        print('year', 'articles', 'neg', 'neu', 'pos',
              'entropy', 'entropy_sd', sep='\t', file=fout)
        for y in sorted(yearly):
            ydata = yearly[y]
            c = Counter(ydata['polarity'])
#            print(y + '-06-01', ydata['articles'], c['neg'], c['neu'], c['pos'],
            print(y, ydata['articles'], c['neg'], c['neu'], c['pos'],
                  np.mean(ydata['entropy']), np.std(ydata['entropy']),
                  sep='\t', file=fout)
    with open(f'boxplot-{j}.txt', 'wt') as fout:
        for pol in ('neg', 'neu', 'pos'):
            pd = [x[pol] for x in perart.values()]
            bp = plt.boxplot(pd)
            w1 = bp['whiskers'][0].get_ydata()[1]
            w2 = bp['whiskers'][1].get_ydata()[1]
            b1 = bp['boxes'][0].get_ydata()[0]
            b2 = bp['boxes'][0].get_ydata()[2]
            median = bp['medians'][0].get_ydata()[0]
            outliers = bp['fliers'][0].get_ydata()
            print(j, pol, w1, w2, b1, b2, median, len(outliers))
            print("%", j, pol, w1, w2, b1, b2, median, file=fout)
            print(f"\\addplot+[{pol}, boxplot prepared={{"
                 f"lower whisker={w1}, lower quartile={b1},"
                 f"median={median},upper quartile={b2}, upper whisker={w2},"
                  "},] coordinates {", file=fout)
            for o in outliers:
                print(f"(0,{o})", end=" ", file=fout)
            print("};", file=fout)
        with open(f'perarticle-{j}.txt', 'wt') as fout:
            print(f"id\tyear\tmonth\tdate\tneg\tneu\tpos", file=fout)
            for aid, adata in perart.items():
                yy, mm, dd = adata['date'].split('-')
                print(f"{aid}\t{yy}\t{mm}\t{dd}\t{adata['neg']}\t{adata['neu']}\t{adata['pos']}", file=fout)       
    print()
