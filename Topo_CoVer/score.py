import torch
import dgl


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def eval_or_test_model(model, paras):
    results = [] 
    model.eval()
            
    hg_list = []
    for item in paras:
        hg = init_graph(len(item))
        hg_list.append(hg)

    bg = dgl.batch(hg_list)

    with torch.no_grad():
        out = model(bg, paras)
        out = torch.sigmoid(out)

        for o in out:
            results.append(o)
            
    return results



# ##########################################################################################


def global_coherence_score(model, paras):

    model.eval()
    results = eval_or_test_model(model, paras)
    return results

