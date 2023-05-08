import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import dgl
import dgl.nn as dglnn
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from modified_transformers.modeling_roberta import RobertaForSequenceClassificationWoPositional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Code refers to https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero
class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        rel_names,
        num_bases,
        *,
        weight=True,
        bias=True,
        activation=None,
        self_loop=False,
        dropout=0.0
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis(
                    (in_feat, out_feat), num_bases, len(self.rel_names)
                )
            else:
                self.weight = nn.Parameter(
                    torch.Tensor(len(self.rel_names), in_feat, out_feat)
                )
                nn.init.xavier_uniform_(
                    self.weight, gain=nn.init.calculate_gain("relu")
                )

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {
                self.rel_names[i]: {"weight": w.squeeze(0)}
                for i, w in enumerate(torch.split(weight, 1, dim=0))
            }
        else:
            wdict = {}

        if g.is_block:
            inputs_src = inputs
            inputs_dst = {
                k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
            }
        else:
            inputs_src = inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs_dst[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}



class TransformerModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    def forward(self, sentences):
        max_len = 512
        
        if len(sentences) <= 40:
            batch = self.tokenizer(sentences, padding=True, return_tensors="pt")
            input_ids = batch['input_ids'][:, :max_len].to(DEVICE)
            attention_mask = batch['attention_mask'][:, :max_len].to(DEVICE)
            output = self.model(input_ids, attention_mask, output_hidden_states=True)
            embeddings = output['hidden_states'][-1][:, 0, :]
        else:
            embeddings = []
            batch_size = 40
            for k in range(0, len(sentences), batch_size):
                batch = self.tokenizer(sentences[k:k+batch_size], padding=True, return_tensors="pt")
                input_ids = batch['input_ids'][:, :max_len].to(DEVICE)
                attention_mask = batch['attention_mask'][:, :max_len].to(DEVICE)
                output = self.model(input_ids, attention_mask, output_hidden_states=True)
                embeddings.append(output['hidden_states'][-1][:, 0, :])
            embeddings = torch.cat(embeddings)
        return embeddings

    
class DocumentTransformerModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        
        self.model = RobertaForSequenceClassificationWoPositional.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def forward(self, sentences):
        max_len = 512
        original, mask = [], []
        
        for item in sentences:
            t1 = [item for sublist in [self.tokenizer(sent)["input_ids"] for sent in item] for item in sublist]
            original.append(torch.tensor(t1)); mask.append(torch.tensor([1]*len(t1)))

        original = pad_sequence(original, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:, :max_len].to(DEVICE)
        mask = pad_sequence(mask, batch_first=True, padding_value=0)[:, :max_len].to(DEVICE)
        output = self.model(original, mask, output_hidden_states=True)
        embeddings = output['hidden_states'][-1][:, 0, :]
        return embeddings


    
class GraphNetwork(nn.Module):
    def __init__(self, hidden_features, out_features, rel_types, readout):
        super().__init__()
            
        self.in_features = 768
        self.hidden_features = hidden_features

        self.transformer = TransformerModel('../deberta-base')
        self.document_transformer = DocumentTransformerModel('../deberta-base')
        self.readout = readout

        self.gcn1 = RelGraphConvLayer(self.in_features, hidden_features, rel_types, activation=F.relu, self_loop=True, num_bases=2)
        self.gcn2 = RelGraphConvLayer(hidden_features, out_features, rel_types, activation=F.relu, self_loop=True, num_bases=2)
        self.scorer = nn.Linear(out_features, 1)
        
    def forward(self, bg, sentences):
        
        all_sentences = [sent for instance in sentences for sent in instance]
        sentence_embed = self.transformer(all_sentences)
        document_embed = self.document_transformer(sentences)

        bg.nodes['sent'].data['feat'] = sentence_embed
        bg.nodes['doc'].data['feat'] = document_embed

        hidden = self.gcn1(bg, bg.ndata['feat'])    
        hidden = self.gcn2(bg, hidden)

        with bg.local_scope():
            bg.ndata['h'] = hidden

            result = 0
            for ntype in hidden.keys():
                if self.readout == 'sum':
                    result = result + dgl.sum_nodes(bg, 'h', ntype=ntype)
                elif self.readout == 'max':
                    result = result + dgl.max_nodes(bg, 'h', ntype=ntype)

            y = self.scorer(result)
            return y
    
