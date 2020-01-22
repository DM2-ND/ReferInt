import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
torch.backends.cudnn.enabled=False


class GlobalAttention(nn.Module):
    ''' class of global attention '''
    def __init__(self, hidden_dim, inter_dim, device):
        super().__init__()

        self.device = device
        self.projection = nn.Sequential(nn.Linear(hidden_dim, inter_dim),
                                        nn.ReLU(True),
                                        nn.Linear(inter_dim, 1)).to(self.device)

    def forward(self, encoder_outputs):

        ''' calculate attention weights '''
        # (Batch, Module, Hidden) -> (Batch , Module, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)

        ''' get attentioned vector '''
        # (Batch, Module, Hidden) * (Batch, Module, 1) -> (Batch, Hidden)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class LocalAttention(nn.Module):
    ''' class of local attention '''
    def __init__(self, hidden_dim, inter_dim, device):
        super().__init__()

        self.device = device
        self.projection = nn.Sequential(nn.Linear(hidden_dim, inter_dim),
                                        nn.ReLU(True),
                                        nn.Linear(inter_dim, 1)).to(self.device)

    def forward(self, encoder_outputs):

        ''' calculate attention weights '''
        # (Batch, Length, Hidden) -> (Batch , Length, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)

        ''' get attentioned vector '''
        # (Batch, Length, Hidden) * (Batch, Length, 1) -> (Batch, Hidden)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class InteractiveAttn(torch.nn.Module):

    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, device, maxlen,
                    nodeWeight, node_dim, intermediate_size):
        super(InteractiveAttn, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.inter_size = intermediate_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.node_dim = node_dim
        self.bidirectional = bidirectional
        self.device = device
        self.layer_size = 2 if self.bidirectional else 1
        self._d = 3

        self.WordEmbedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.WordEmbedding.weight.data.uniform_(-1., 1.)
        self.WordEmbedding.weight.requires_grad = True

        self.NodeEmbedding = nn.Embedding.from_pretrained(nodeWeight)
        self.NodeEmbedding.weight.requires_grad = True

        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional,
                            batch_first=True)

        self.lcoalAttnGC = LocalAttention(self.layer_size * self.hidden_size,
                                        self.inter_size,
                                        self.device).to(self.device)

        self.lcoalAttnRC = LocalAttention(self.layer_size * self.hidden_size,
                                        self.inter_size,
                                        self.device).to(self.device)

        self.globalAttn = GlobalAttention(self.layer_size * self.hidden_size,
                                          self.inter_size,
                                          self.device).to(self.device)

        self.out_from_global = nn.Sequential(nn.Linear(self.layer_size * self.hidden_size, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.output_size))

        self.weight = nn.Sequential(nn.Linear(2 * self.layer_size * self.hidden_size, self.inter_size),
                                      nn.ReLU(True),
                                      nn.Linear(self.inter_size, 1),
                                      nn.Sigmoid()).to(self.device)


    ''' ncf for neighbor former, lc for local context, ncl for neighbor latter, rc for referred context '''
    ''' netcing for network context in citing paper, netced for network context in cited paper '''
    def forward(self, ncf, lc, ncl, rc, netcing, netced):

        ''' dataloader provide numpy, convert into torch tensor'''
        ncf = Variable(torch.from_numpy(ncf).to(self.device))
        lc = Variable(torch.from_numpy(lc).to(self.device))
        ncl = Variable(torch.from_numpy(ncl).to(self.device))
        rc = Variable(torch.from_numpy(rc).to(self.device))
        netc_citing = Variable(torch.from_numpy(netcing).to(self.device))
        netc_cited = Variable(torch.from_numpy(netced).to(self.device))

        ''' language model encoder for local representation '''
        ncf_emb = self.WordEmbedding(ncf)
        ncf_out, (_, _) = self.lstm(ncf_emb)
        ncf_attn = self.lcoalAttnGC(ncf_out)[0]

        lc_emb = self.WordEmbedding(lc)
        lc_out, (_, _) = self.lstm(lc_emb)
        lc_attn = self.lcoalAttnGC(lc_out)[0]

        ncl_emb = self.WordEmbedding(ncl)
        ncl_out, (_, _) = self.lstm(ncl_emb)
        ncl_attn = self.lcoalAttnGC(ncl_out)[0]

        rc_emb = self.WordEmbedding(rc)
        rc_out, (_, _) = self.lstm(rc_emb)
        rc_attn = self.lcoalAttnRC(rc_out)[0]

        netc_citing_emb = self.NodeEmbedding(netc_citing)
        netc_citing_out = F.normalize(torch.sum(netc_citing_emb, dim=1), p=2, dim=1)
        netc_cited_emb = self.NodeEmbedding(netc_cited)
        netc_cited_out = F.normalize(torch.sum(netc_cited_emb, dim=1), p=2, dim=1)
        netc_out = torch.cat([netc_citing_out, netc_cited_out], dim=1).float()

        ''' set up interactive vector '''
        interactive = torch.zeros(self.batch_size, self._d, self.layer_size * self.hidden_size).to(self.device)
        interactive[:, 0, :] = lc_attn + self.weight(torch.cat((ncf_attn, lc_attn), dim=1)) * ncf_attn
        interactive[:, 1, :] = lc_attn + self.weight(torch.cat((ncl_attn, lc_attn), dim=1)) * ncl_attn
        interactive[:, 2, :] = ncl_attn + self.weight(torch.cat((ncf_attn, ncl_attn), dim=1)) * ncl_attn
        interactive[:, 3, :] = lc_attn + self.weight(torch.cat((netc_out, lc_attn), dim=1)) * netc_out
        interactive[:, 4, :] = lc_attn + self.weight(torch.cat((rc_attn, lc_attn), dim=1)) * rc_attn
        interactive[:, 5, :] = netc_out + self.weight(torch.cat((rc_attn, netc_out), dim=1)) * rc_attn

        ''' global attention for referital behavior embedding '''
        assert interactive.shape == torch.Size([self.batch_size, self._d, self.layer_size * self.hidden_size])

        global_attn = self.globalAttn(interactive)[0]
        logits = self.out_from_global(global_attn)

        return logits