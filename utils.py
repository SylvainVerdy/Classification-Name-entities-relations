import json
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import torch.optim as optim
import torch
from models import *
import numpy as np
import os

device = torch.device('cuda' if 'cuda:0' and torch.cuda.is_available() else 'cpu')

def search_entity(sentence):
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = word_tokenize(sentence)
    #sentence = [word for word in sentence if not word in stopwords.words()]
    sentence = ' '.join(sentence)
    sentence = sentence.replace('< e1 >', '<e1>')
    sentence = sentence.replace('< e2 >', '<e2>')
    sentence = sentence.replace('< /e1 >', '</e1>')
    sentence = sentence.replace('< /e2 >', '</e2>')
    sentence = sentence.split()
    return sentence

def convert(df):
    comment, sentencev2 = [], [] 
    for i, row in df.iterrows():
        sentence = row['sentence']
        sentence = search_entity(sentence)
        sentence_2 = " ".join(sentence).lower()
        comment.append(sentence)
        sentencev2.append(sentence_2)

    df['comment'] = comment
    df['sentencev2'] = sentencev2
    return df

def collate_fn(batch):
    '''
    padding custom tokenizer 
    '''
    max_len = max([len(b[0]) for b in batch])
    max_len_att = max([len(b[1]) for b in batch])
    input_ids = [torch.cat((b[0] , torch.tensor(([3]) * (max_len - len(b[0])))))  for b in batch]
    attention_mask = [torch.cat((b[1] , torch.tensor(([0]) * (max_len_att - len(b[1])))))  for b in batch]
    labels = [b[2] for b in batch]
    return torch.stack(input_ids).long(), torch.stack(attention_mask).long() , torch.stack(labels)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def accuracy(preds, targets):
    return (torch.sum((torch.argmax(preds,dim=-1) == targets)) / len(targets))* 100
    
def load_model(model_name=None):
  model = None
  model_name = 'LittleEncoder'
  if not os.path.exists( './model_save/LittleEncoder.ckpt'):
      raise Exception("Model doesn't exists! Train first!")
  try:
      if model_name == 'LittleEncoder':
          model = Model().to(device)
      model.load_state_dict(torch.load('./model_save/LittleEncoder.ckpt'))
      model.eval()
      print("***** Model Loaded *****")
  except:
      raise Exception("Some model files might be missing...")
  return model


def save_model(model, optimizer, epoch, loss):
  print('Saving model!')
  try:
    torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss,
              }, f'./model_save/LittleEncoder_{epoch}.ckpt')
  except:
    raise Exception('Saving problems!')



def dist_loss(teacher, student, Temperature=0.1):
    prob_t = F.softmax(teacher/Temperature, dim=1)
    log_prob_s = F.log_softmax(student/Temperature, dim=1)
    dist_loss = -(prob_t*log_prob_s).sum(dim=1).mean()
    return dist_loss

def loss_func( output, bert_prob, real_label, weights, config):
    alpha = config['alpha']
    criterion_mse = torch.nn.MSELoss()
    criterion_ce = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights).type(torch.float32).to(device))
    loss = alpha*criterion_ce(output, real_label) + (1-alpha) * dist_loss(bert_prob.detach(), output)
    return loss