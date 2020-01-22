import sys
import os
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)
sys.path.append(BASE_DIR)

import pickle
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
torch.backends.cudnn.enabled=False

lr = 0.001
epochs = 100
batch_size = 32
seed = 666
dropout = 0.5
embed_dim = 200
hidden_size = 100
weight_decay = 0.001
intermediate_size = 100
cuda_able = True
bidirectional = True
torch.manual_seed(seed)

npath = BASE_DIR + '/data/node_embs.npy'
data = BASE_DIR + '/data/academic_corpus'

node_weight = torch.from_numpy(np.load(npath))
node_dim = node_weight.shape[1]

device = torch.device('cuda') if torch.cuda.is_available() and cuda_able else torch.device('cpu')
print('Cuda status is {}'.format(device))

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics


def get_evaluation_results(y_true, y_pred):

    macro_precision = round(precision_score(y_true, y_pred, average='macro'), 4)
    macro_recall = round(recall_score(y_true, y_pred, average='macro'), 4)
    micro_f1 = round(f1_score(y_true, y_pred, average='micro'), 4)
    macro_f1 = round(f1_score(y_true, y_pred, average='macro'), 4)

    return micro_f1, macro_precision, macro_recall, macro_f1,


from dataloder import DataLoader
import model as model

data = torch.load(data)
maxlen = data["maxlen"]
vocab = data['vocab']
vocab_size = len(vocab)
out = data['label']
out_size = len(out)

trainSet = DataLoader(data['train']['ncf'],
                      data['train']['lc'],
                      data['train']['ncl'],
                      data['train']['rc'],
                      data['train']['netcing'],
                      data['train']['netced'],
                      data['train']['label'],
                      maxlen=maxlen,
                      batch_size=batch_size,
                      device=device)

testSet = DataLoader(data['test']['ncf'],
                     data['test']['lc'],
                     data['test']['ncl'],
                     data['test']['rc'],
                     data['test']['netcing'],
                     data['test']['netced'],
                     data['test']['label'],
                     maxlen=maxlen,
                     batch_size=batch_size,
                     shuffle=False,
                     device=device)

model = model.InteractiveAttn(batch_size=batch_size,
                                output_size=out_size,
                                hidden_size=hidden_size,
                                vocab_size=vocab_size,
                                embed_dim=embed_dim,
                                bidirectional=bidirectional,
                                dropout=dropout,
                                device=device,
                                maxlen=maxlen,
                                nodeWeight=node_weight,
                                node_dim=node_dim,
                                intermediate_size=intermediate_size
                                ).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss().to(device)



def evaluate():

    model.eval()

    corrects, eval_loss = 0, 0
    y_pred, y_true = [], []

    for nca, lc, ncf, rc, netcing, netced, label in tqdm(testSet, desc='Evaluate', leave=False):

        pred = model(nca, lc, ncf, rc, netcing, netced)

        label = torch.from_numpy(label).reshape(-1).to(device)
        loss = criterion(pred, label)

        eval_loss += loss.item()
        corrects += (torch.max(pred, 1)[1].view(label.size()).data == label.data).sum()

        y_pred += torch.max(pred, 1)[1].view(label.size()).data.cpu().numpy().tolist()
        y_true += label.data.cpu().numpy().tolist()

    evaluation = get_evaluation_results(np.array(y_true), np.array(y_pred))

    return loss, evaluation


def train():

    model.train()

    total_loss = 0
    for ncf, lc, ncl, rc, netcing, netced, label in tqdm(trainSet, desc='Train', leave=False):

        optimizer.zero_grad()
        target = model(ncf, lc, ncl, rc, netcing, netced)

        label = torch.from_numpy(label).reshape(-1).to(device)
        loss = criterion(target, label)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


total_start_time = time.time()

try:

    evaluations = []
    train_loss = []

    for epoch in range(1, epochs+1):

        epoch_start_time = time.time()
        loss = train()
        train_loss.append(loss)

        print('start of epoch {:3d}'.format(epoch))
        print('loss on train {:5.6f}'.format(loss))

        loss, evaluation = evaluate()
        print('dev evaluation {}'.format(evaluation))
        evaluations.append(evaluation)

except KeyboardInterrupt:
    print("-"*90)
    print("Exiting training early | Time: {:5.2f}min".format((time.time() - total_start_time)/60.0))
