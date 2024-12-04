#!/usr/bin/env python3

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as prfs
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader
from transformers import AdamW
import traceback


#def read_data(fname='concit.tsv', sep=' [SEP] '):
def read_data(fname='concit.tsv', sep=' | '):
    texts, labels, labelset = [], [], set()
    with open(fname, 'rt') as fp:
        csvr = csv.DictReader(fp, delimiter='\t')
        for row in csvr:
            texts.append(sep.join((row['citation'], row['context'])))
            labels.append(row['polarity'])
            labelset.add(row['polarity'])
    labelset = sorted(labelset)
    labelmap = {k:labelset.index(k) for k in labelset}
    labels = [labelmap[k] for k in labels]

    return texts, labels, labelmap


class CitDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer=None):
        self.labels = labels
        if tokenizer:
            self.texts = tokenizer(texts,
                truncation=True, padding='longest', max_length=256)
        else:
            self.texts = texts

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.texts.items()}
        item['labels'] = torch.tensor(self.labels[i])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)

    p, r, f, _ = prfs(labels, pred, average='macro')
    return {'precision': p, 'recall': r, 'f1': f}

if __name__ == "__main__":
    models = {
            "allenai/scibert_scivocab_uncased": "scibert-uncased",
            "roberta-base": "roberta-base",
    }

    texts, labels, labelmap = read_data()
    txttrn, txtval, labtrn, labval = train_test_split(
            texts, labels, test_size=100)

    eval_batch_size = 4
    epochs = 20
    lr = 1e-05
    batch_size = 4
    split=0
    mname = "roberta-base"

    mshort = models[mname]
    try:
        print(f"{mshort}-{batch_size}-{lr}-{split}")
        output_dir = f"./results-{mshort}-{batch_size}-{lr}-{split}"
        if os.path.exists(output_dir):
            print(f"{output_dir} exists.")
        tokenizer = AutoTokenizer.from_pretrained(mname)
        trndata = CitDataset(txttrn, labtrn, tokenizer=tokenizer)
        valdata = CitDataset(txtval, labval, tokenizer=tokenizer)
        m = AutoModelForSequenceClassification.from_pretrained(mname,
            num_labels=len(labelmap))

        mshort = models[mname]
        training_args = TrainingArguments(
            output_dir = output_dir,
            num_train_epochs = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = eval_batch_size,
            warmup_steps = 500,
            weight_decay = 0,
            learning_rate = lr,
            logging_dir=f'./logs-{mshort}-{batch_size}-{lr}-{split}',
            logging_steps=500,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            save_total_limit=2,
            metric_for_best_model='eval_f1',
        )

        trainer = Trainer(
            model=m,
            args=training_args,
            train_dataset=trndata,
            eval_dataset=valdata,
            compute_metrics=compute_metrics,
        )
        trainer.train()
    except:
        print(traceback.format_exc())
        print(f"Failed: {mshort}-{batch_size}-{lr}-{split}")

tokenizer.save_pretrained("final-model")
trainer.save_model("final-model")
