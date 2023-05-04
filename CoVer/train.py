import argparse, time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import dgl
import os

from model import GraphNetwork
from dataloader import load_dataset
from transformers.optimization import AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ##########################################################################################
# Optimizer and Scheduler Setting
class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

def configure_transformer_optimizer(model, args):
    "Prepare AdamW optimizer for transformer encoders"
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    decay_parameters = [name for name in decay_parameters if ("bias" not in name and 'gcn' not in name and 'scorer' not in name)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]   
    optimizer_kwargs = {
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "lr": args.lr
    } 
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

def configure_gcn_optimizer(model, args):
    "Prepare Adam optimizer for GCN decoders"
    optimizer = optim.Adam([
        {'params': model.gcn1.parameters()},
        {'params': model.gcn2.parameters()},
        {'params': model.scorer.parameters()}
    ], lr=args.lr0, weight_decay=args.wd0)
    return optimizer

def configure_scheduler(optimizer, num_training_steps, args):
    "Prepare scheduler"
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )    
    return lr_scheduler
# ##########################################################################################


def configure_dataloaders(dataset, batch_size):
    "Prepare dataloaders"
    train_loader = load_dataset(f'../data/hier/{dataset}/train.lower', batch_size, shuffle=True)    
    valid_loader = load_dataset(f'../data/hier/{dataset}/val.lower', batch_size, shuffle=True)
    test_loader = load_dataset(f'../data/hier/{dataset}/test.lower', batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

# ##########################################################################################

# Initiate Single Heterogeneous Graph
def init_graph(sent_cnt):
    '''
    Edge relations:
    0: Sequential edges: edges between adjacent sentences
    1: Skip edges: edges between nonadjacent sentences
    2: para-to-sentence edges
    '''

    graph_data = {('sent', '0', 'sent'):([], []), ('sent', '1', 'sent'):([], []), ('doc', '2', 'sent'):([], [])}
        

    for i in range(sent_cnt):
        if i < sent_cnt - 1:
            graph_data[('sent', '0', 'sent')][0].append(i)
            graph_data[('sent', '0', 'sent')][1].append(i+1)

        for j in range(i+2, sent_cnt):
            graph_data[('sent', '1', 'sent')][0].append(i)
            graph_data[('sent', '1', 'sent')][1].append(j)

        graph_data[('doc', '2', 'sent')][0].append(0)
        graph_data[('doc', '2', 'sent')][1].append(i)
    
        
    hg = dgl.heterograph(graph_data).to(DEVICE)
    return hg


# ##########################################################################################


def train_model(model, dataloader, optimizer=None):
    losses = []
    assert optimizer!=None
    
    model.train()
    
    for pos_sents, neg_sents, label in tqdm(dataloader, leave=False):
        
        optimizer.zero_grad()

        pos_hg_list = []
        for item in pos_sents:
            hg = init_graph(len(item))
            pos_hg_list.append(hg)

        pos_bg = dgl.batch(pos_hg_list)

        pos_out = model(pos_bg, pos_sents)


        neg_hg_list = []
        for item in neg_sents:
            hg = init_graph(len(item))
            neg_hg_list.append(hg)

        neg_bg = dgl.batch(neg_hg_list)

        neg_out = model(neg_bg, neg_sents)

        out = torch.cat([pos_out, neg_out], dim=1)
        out = torch.softmax(out, dim=1)
        pos_out = out[:,0]
        neg_out = out[:,1]
        
        label = torch.tensor(label).float().to(DEVICE)
        loss = loss_function(pos_out, neg_out, label)       
        
        loss.backward()
        optimizer.step()
            
        losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    return avg_loss




def eval_or_test_model(model, dataloader):
    losses, results = [], []    
    model.eval()
    
    acc, doc_sum = 0, 0
    for pos_sents, neg_sents, label in tqdm(dataloader, leave=False):
        
        pos_hg_list = []
        for item in pos_sents:
            hg = init_graph(len(item))
            pos_hg_list.append(hg)

        pos_bg = dgl.batch(pos_hg_list)


        neg_hg_list = []
        for item in neg_sents:
            hg = init_graph(len(item))
            neg_hg_list.append(hg)

        neg_bg = dgl.batch(neg_hg_list)

        with torch.no_grad():
            pos_out = model(pos_bg, pos_sents)
            neg_out = model(neg_bg, neg_sents)

            out = torch.cat([pos_out, neg_out], dim=1)
            out = torch.softmax(out, dim=1)
            pos_out = out[:,0]
            neg_out = out[:,1]

            for sent_id, (pos_o, neg_o, l) in enumerate(zip(pos_out, neg_out, label)):
                doc_sum += 1
                results.append((pos_sents[sent_id], neg_sents[sent_id], pos_o, neg_o))

                if pos_o > neg_o and l == 1:
                    acc += 1
                elif pos_o <= neg_o and l == -1:
                    acc += 1

            label = torch.tensor(label).float().to(DEVICE)
            loss = loss_function(pos_out, neg_out, label)
            losses.append(loss.item())

    avg_loss = round(np.mean(losses), 4)
    return avg_loss, acc/doc_sum, results




# ##########################################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate for transformers.")
    parser.add_argument("--lr0", type=float, default=1e-4, help="Learning rate for GCN.")
    parser.add_argument("--wd", default=0.0, type=float, help="Weight decay for transformers.")
    parser.add_argument("--wd0", default=1e-6, type=float, help="Weight decay for GCN.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
    parser.add_argument("--adam-beta1", default=0.9, type=float, help="beta1 for AdamW optimizer.")
    parser.add_argument("--adam-beta2", default=0.999, type=float, help="beta2 for AdamW optimizer.")
    parser.add_argument("--lr-scheduler-type", default="linear")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Steps used for a linear warmup from 0 to lr.")
    parser.add_argument("--warmup-ratio", type=float, default=0.0, help="Ratio of total training steps used for a linear warmup from 0 to lr.")

    parser.add_argument("--dataset", default="roc", help="Which dataset: roc, nips, sind, aan")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs.")
    parser.add_argument("--hdim", type=int, default=200, help="Hidden dim GCN.")

    parser.add_argument("--readout", type=str, default='sum', help="Graph classification readout function")
    parser.add_argument("--margin", type=float, default=0.1, help="Loss function margin parameter")


    global args
    args = parser.parse_args()
    print(args)

    run_ID = int(time.time())
    print ('run id:', run_ID)

    # define graph relation types
    rel_types = [str(i) for i in range(3)]
    model = GraphNetwork(args.hdim, args.hdim, rel_types, args.readout).to(DEVICE)
    
    global loss_function
    loss_function = torch.nn.MarginRankingLoss(args.margin).to(DEVICE)

    optimizer1 = configure_transformer_optimizer(model, args)
    optimizer2 = configure_gcn_optimizer(model, args)
    optimizer = MultipleOptimizer(optimizer1, optimizer2)

    train_loader, valid_loader, test_loader = configure_dataloaders(args.dataset, args.batch_size)


    best_acc, best_loss = None, None
    best_epoch = None
    for e in range(args.epochs):
        
        train_loss = train_model(model, train_loader, optimizer)
        valid_loss, valid_acc, valid_results = eval_or_test_model(model, valid_loader)
        test_loss, test_acc, test_results = eval_or_test_model(model, test_loader)
        
        print(f'\nEpoch {e+1}: train loss: {train_loss}, valid loss: {valid_loss} acc: {valid_acc} test loss: {test_loss} acc: {test_acc}\n')
        
        if not os.path.exists(f'../saved/{args.dataset}/'):
            os.makedirs(f'../saved/{args.dataset}/')

        if best_acc == None or best_acc < valid_acc:
            torch.save(model.state_dict(), f'../saved/{args.dataset}/{run_ID}_model.pt')
            best_acc = valid_acc
            best_loss = valid_loss
            best_epoch = e
        elif best_acc == valid_acc and best_loss > valid_loss:
            torch.save(model.state_dict(), f'../saved/{args.dataset}/{run_ID}_model.pt')
            best_loss = valid_loss
            best_epoch = e

    print('\n\n')

    model.load_state_dict(torch.load(f'../saved/{args.dataset}/{run_ID}_model.pt'))
    model.eval()
        
    test_loss, acc, results = eval_or_test_model(model, test_loader, args.sent_seq, args.has_doc)
    print(f'\nEpoch: {best_epoch+1} Test loss, acc at best valid acc: {test_loss}, {acc}')
    
        
    with open(f'results/{args.dataset}/score_{run_ID}.txt', 'w') as f:
        for line in results:
            content = '\t'.join([str(s) for s in line])
            f.write(content + '\n')
