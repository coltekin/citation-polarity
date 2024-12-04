#!/usr/bin/env python3

import argparse
import csv
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification

#mname = "dbmdz/electra-small-turkish-cased-discriminator"

def read_input(fname, context=True):
    texts, ids, dates = [], [], []
    with open(fname, 'rt') as f:
        csvr = csv.reader(f, delimiter='\t')
        _ = next(csvr)
        for row in csvr:
            ids.append(row[0])
            if context:
                texts.append(row[3])
            else:
                texts.append(row[2])
            dates.append(row[1])
    return texts, ids, dates

def get_batch(texts, batchnum, tokenizer, batchsize=32):
    start = batchnum*batchsize
    if start >= len(texts):
        return []
    return tokenizer.batch_encode_plus(texts[start:start+batchsize],
        return_tensors='pt', truncation=True, padding=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('input')
    ap.add_argument('--output', '-o')
    ap.add_argument('--model', '-m', default="coltekin/citation-polarity-roberta-base")
    ap.add_argument('--batchsize', '-b', type=int, default=32)
    args = ap.parse_args()

    if args.output is None:
        args.output = args.input + '.pred'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForSequenceClassification.from_pretrained(
            args.model).to(device)
    config = AutoConfig.from_pretrained(args.model)

    print(f"Reading {args.input}...")
    start = time.time()
    texts, ids, dates = read_input(args.input)
    print(f"Loaded {len(texts)} instances in {time.time() - start} sec.")

    print("Predicting...", flush=True)
    nbatches = len(texts) // args.batchsize
    logits = []
    start, i = time.time(), 0
    x = get_batch(texts, i, tokenizer,
            batchsize=args.batchsize)
    while x:
        x = x.to(device)
        pred = m(**x)
        logits.extend(pred['logits'].tolist())
        if (i % 100) == 0: print(f"\r{i:7d}/{nbatches}", end="", flush=True)
        i += 1
        x = get_batch(texts, i, tokenizer,
            batchsize=args.batchsize)
    print(f'\n{i} batches in {time.time() - start} seconds',
            flush=True)

    with open(args.output, 'wt') as f:
        print("aid\ttimestemp\t" + '\t'.join(
            [config.id2label[i] for i in sorted(config.id2label)]),
            file=f)
        for i, pp in enumerate(logits):
            print(f"{ids[i]}\t{dates[i]}\t" +
                    '\t'.join([str(x) for x in pp]), file=f)
