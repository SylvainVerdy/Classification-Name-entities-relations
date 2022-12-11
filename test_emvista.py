import os
import torch.optim as optim
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import random
from torchinfo import summary
from sklearn.metrics import classification_report
from utils import *
import wandb

wandb.init(project="emvista", entity="sylvainverdy") # too late for this


# cell 2
dataset = load_dataset("sem_eval_2010_task_8")
dataset


label2class = {0: 'Other',
               1: 'Message-Topic(e1,e2)', 2: 'Message-Topic(e2,e1)',
               3: 'Product-Producer(e1,e2)', 4: 'Product-Producer(e2,e1)',
               5: 'Instrument-Agency(e1,e2)', 6: 'Instrument-Agency(e2,e1)',
               7: 'Entity-Destination(e1,e2)', 8: 'Entity-Destination(e2,e1)',
               9: 'Cause-Effect(e1,e2)', 10: 'Cause-Effect(e2,e1)',
               11: 'Component-Whole(e1,e2)', 12: 'Component-Whole(e2,e1)',
               13: 'Entity-Origin(e1,e2)', 14: 'Entity-Origin(e2,e1)',
               15: 'Member-Collection(e1,e2)', 16: 'Member-Collection(e2,e1)',
               17: 'Content-Container(e1,e2)', 18: 'Content-Container(e2,e1)'}

dataset['train'].train_test_split(test_size=0.1)

df = pd.DataFrame(dataset['train'])
df['label'] = [label2class[r] for r in df['relation']]

train, dev = train_test_split(df, test_size=0.2)
test = pd.DataFrame(dataset['test'])
test['label'] = [label2class[r] for r in test['relation']]
test_t = test 

train = convert(train)
dev = convert(dev)
test = convert(test)

# weights class 
weights = (train.label.value_counts()/len(train)).tolist()
print(len(weights))


config = {
    "lr" : 1e-4,
    "epochs" : 30,
    "batch_size" : 32,
    "alpha": 0.5
}

wandb.config = {
  "learning_rate": 1e-4,
  "epochs": 30,
  "batch_size": 32
}

from CustomDataloader import *
from models import *

device = torch.device('cuda' if 'cuda:0' and torch.cuda.is_available() else 'cpu')

train_dataset = CustomDataset(df=train)
dev_dataset = CustomDataset(df=dev)
test_dataset = CustomDataset(df=test)

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=config["batch_size"], num_workers=0, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, collate_fn=collate_fn, batch_size=config["batch_size"], num_workers=0, shuffle=True)
test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=config["batch_size"], num_workers=0, shuffle=True)

from sklearn.utils import class_weight


class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(train.relation),
                                        y = train.relation                                                    
                                    )
class_weights = dict(zip(np.unique(train.relation), class_weights))
weights = list(class_weights.values())



SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if 'cuda:0' and torch.cuda.is_available() else 'cpu')
model = Model().to(device)
criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor(weights).type(torch.float32).to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=0.0000009)
lr_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=100, max_iters=2000)

model_bert = BertForSequenceClassification().to(device)


def fit(train):  
  
  print('Start training ...')
  best_loss = 9e15

  for epoch in range(config['epochs']):
    model.train(True)
    with tqdm(train, unit="batch") as tepoch:
      tepoch.set_description(f"Epoch {epoch}")
      all_loss = 0
      tr_loss = 0
      total, correct = 0,0
      for step, batch in enumerate(tepoch):
        x, mask, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer.zero_grad()
        output = model(x)
        output_bert = model_bert(x)
        loss = loss_func(output, output_bert, targets, weights, config)
        # loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=5)

        optimizer.step()
        tr_loss += loss.item()
        total += batch[0].size(0)

        _ , predicted = output.max(1)
        correct += torch.sum(predicted == targets)
        # print(predicted, targets)
        tepoch.set_postfix({'Train loss': loss.item(), 'Accuracy Train': (100*(correct/total))})
      train_loss = tr_loss / len(train)

      acc = 100 * (correct/total)
      lr_scheduler.step()

      print(f'Train loss : {train_loss}, epoch : {epoch}')
      print(f'Train Accuracy : {acc} %, epoch : {epoch}')
      
      val_loss = evaluate(dev_dataloader, epoch, 'validation')
      if val_loss[0] < best_loss:
        best_loss = val_loss[0]
        save_model(model, optimizer, epoch, best_loss)
  print('End training')
  test_loss, acc_test = evaluate(test_dataloader, epoch, 'test')
  print('test ACC:', acc_test)

def evaluate(dataset_eval, epoch, type_eval):
  model.eval()
  print('Start eval ...')

  all_loss = 0
  val_loss, val_epoch_loss = 0,0
  total, correct = 0,0
  best_loss = 0
  y_pred, y_tgt = [], []
  with torch.no_grad():

    for step, batch in enumerate(dataset_eval):
      x, mask, targets = batch[0].to(device), batch[1].to(device), batch[2].to(device)
      output = model(x)
      output_bert = model_bert(x)
      loss = loss_func(output, output_bert, targets, weights, config)
      #loss = criterion(output, targets)
      val_loss += loss.item()
      total += batch[0].size(0)
      _ , predicted = output.max(1)
      correct += torch.sum(predicted == targets)
      y_pred.extend(predicted.detach().cpu().numpy())
      y_tgt.extend(targets.detach().cpu().numpy())
    
    val_epoch_loss = val_loss / len(dataset)

    acc = 100 * (correct/total)
    if type_eval == 'test':
      print(classification_report(y_tgt, y_pred)) #, target_names=list(label2class.values()))
  
    print(f'{type_eval}  loss : {val_epoch_loss}, epoch : {epoch}')
    print(f'{type_eval}  Accuracy : {acc} %, epoch : {epoch}')
  
  return val_epoch_loss, acc

fit(train_dataloader)