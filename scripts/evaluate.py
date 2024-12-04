#!/usr/bin/env python3

import argparse
import csv
import json
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def read_input(fname, context=True):
    d = []
    with open(fname, 'rt') as f:
        csvr = csv.DictReader(f, delimiter='\t')
        for row in csvr:
            d.append(row)
    return d

def get_batch(texts, batchnum, tokenizer, batchsize=32):
    start = batchnum*batchsize
    if start >= len(texts):
        return []
    return tokenizer.batch_encode_plus(texts[start:start+batchsize],
        return_tensors='pt', truncation=True, padding=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('gold')
    ap.add_argument('pred')
    args = ap.parse_args()
    labels = np.array(['neg', 'neu', 'pos'])


    gold = read_input(args.gold)
    pred = read_input(args.pred)

#    for i,(a, b, c) in enumerate([[x[l] for l in labels] for x in pred]):
#        print(i, gold[i]['citation'], a, b, c)
    logits = np.array([[float(x[l]) for l in labels] for x in pred])
    pred_labels = labels[np.argmax(logits, axis=1)]
    gold_labels = [x['polarity'] for x in gold]
    print(classification_report(gold_labels, pred_labels))
