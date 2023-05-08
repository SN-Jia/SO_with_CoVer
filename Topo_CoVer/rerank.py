import numpy as np
from tqdm import tqdm
import torch
import argparse, itertools

from topological_sort import Stats, Graph
from model import GraphNetwork
from score import global_coherence_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# get score matrix
def arrange_data(sentences, prob):
    result = []
    total_cnt = 0
    for para in sentences:
        tmp_dic = {}
        permu_id = 0
        sent_num = len(para)
        for i in range(sent_num):
            for j in range(i+1, sent_num):
                key1 = (j, i)
                key2 = (i, j)
                tmp_dic[permu_id] = [key1, key2]
                permu_id += 1

        cnt = sent_num * (sent_num - 1) // 2
        tmp_result = [[0]*sent_num for _ in range(sent_num)]
        for row_id in range(cnt):
            index = torch.argmax(prob[total_cnt+row_id])
            sent1, sent2 = tmp_dic[row_id][index]
            #print(f'pair: {sent1} {sent2} {prob[total_cnt+row_id][index]}')
            tmp_result[sent1][sent2] = prob[total_cnt+row_id][index]
            tmp_result[sent2][sent1] = prob[total_cnt+row_id][index^1]


        result.append(tmp_result)
        total_cnt += cnt

    return result


# ##########################################################################################


def validateConfidence(graph_model, paras, candidate_sents, pred_order, res_order):
    print(f'pre: {pred_order} candidate: {candidate_sents} res: {res_order}')
    candidates = list(itertools.permutations(candidate_sents, len(candidate_sents)))
    print(f'cand: {candidates}')
    orders = [pred_order + list(cand) + res_order for cand in candidates]
    print(f'orders: {[order for order in orders]}')

    texts = [[paras[i] for i in order] for order in orders]

    graph_score = global_coherence_score(graph_model, texts)

    score = [x for x in graph_score]
    print(f'total score: {score}')
    score = torch.tensor(score).float()
    id = torch.argmax(score)

    return candidates[id][-1], pred_order + list(candidates[id])
    


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

    parser.add_argument("--dataset", default="roc", help="Which dataset: roc, nips, nsf, sind, aan")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs.")
    parser.add_argument("--hdim", type=int, default=200, help="Hidden dim GCN.")

    parser.add_argument("--score-runID", type=int, default=0, help="Trained score model ID")

    # score model parameter
    parser.add_argument("--readout", type=str, default='sum', help="Graph classification readout function")


    global args
    args = parser.parse_args()
    print(args)


    # ##########################################################################################

    sentences = [] # text
    with open(f'../data/so/{args.dataset}/test.lower') as f:
        for line in f:
            line = line.strip().split(' @eos@ ')
            if len(line) < 2:
                continue
            sentences.append(line)

    score = np.load(f'pairwise_score/roc.npy')
    score = torch.tensor(score)

    score_mat = arrange_data(sentences, score) # score matrix

    # ##########################################################################################

    global loss_function
    global directed

    rel_types = [str(i) for i in range(3)]

    graph_model = GraphNetwork(args.hdim, args.hdim, rel_types, args.readout).to(DEVICE)
    graph_model.load_state_dict(torch.load(f'../saved/{args.dataset}/{args.score_runID}_model.pt', map_location=DEVICE))

    # ##########################################################################################

    print('Sorting and Scoring')

    total_pair_cnt = 0
    orders = []

    for doc_id, (paras, pair_info) in enumerate(tqdm(zip(sentences, score_mat))):
        print(f'--------------------------------{doc_id}--------------------------------')
        
        nvert = len(paras)
        # if nvert < 8:
        #    continue
        npairs = nvert * (nvert-1) // 2

        g = Graph(nvert)

        for i in range(nvert):
            for j in range(i+1, nvert):
                if pair_info[i][j] > pair_info[j][i]:
                    g.addEdge(i, j)
                else:
                    g.addEdge(j, i)

        # g.printGraph()
        order = []
        # Sorting and Scoring
        sent_a = g.findFirstSentence()
        print(f'sent_a: {sent_a}')

        # has cycle
        if sent_a == None:
            g.isCyclic()
            cycle_pos = g.getCyclePos()
            for node in cycle_pos:
                g.removeEdge(node)
            
            next_sent = g.findFirstSentence(verified=False)
            if next_sent == None:
                res_order = []
            else:
                res_order = [next_sent]
            sent_a, order = validateConfidence(graph_model, paras, cycle_pos, [], res_order)
            cnt = nvert - len(cycle_pos)
        else:
            g.removeEdge(sent_a)
            order.append(sent_a)

            cnt = nvert - 1

        while cnt > 0:
            sent_b = g.findFirstSentence()
            print(f'sent_b: {sent_b}')
            # has cycle
            if sent_b == None:
                g.isCyclic()
                cycle_pos = g.getCyclePos()
                for node in cycle_pos:
                    g.removeEdge(node)
                
                next_sent = g.findFirstSentence(verified=False)
                if next_sent == None:
                    res_order = []
                else:
                    res_order = [next_sent]
                sent_a, order = validateConfidence(graph_model, paras, cycle_pos, order, res_order)
                print(f'order: {order}')
                cnt -= len(cycle_pos)
            # no cycle
            else:
                g.removeEdge(sent_b)
                
                if abs(pair_info[sent_a][sent_b] - pair_info[sent_b][sent_a]) <= args.threshold:
                    next_sent = g.findFirstSentence(verified=False)
                    if next_sent == None:
                        res_order = []
                    else:
                        res_order = [next_sent]

                    order = order[:-1]

                    sent_a, order = validateConfidence(graph_model, paras, [sent_a, sent_b], order, res_order)
                    # order += rank_order
                    print(f'order: {order}')

                    
                    for i in range(len(order)-2, 0, -1):
                        sent_m = order[i]
                        sent_n = order[i-1]
                        if abs(pair_info[sent_m][sent_n] - pair_info[sent_n][sent_m]) <= args.threshold:
                            front_order = order[:i-1]
                            latter_order = order[i+1:]
                            _, order = validateConfidence(graph_model, paras, [sent_m, sent_n], front_order, [order[i+1]])
                            order += latter_order
                            print(f'order: {order}')

                else:
                    sent_a = sent_b
                    order.append(sent_b)

                cnt -= 1
            print(f'order: {order}')

        assert len(order) == len(paras)
        orders.append(order)
        # print(orders)

    # ##########################################################################################

    # final result
    stats = Stats()
    
    for index, item in enumerate(orders):
        print(f'{index}\t{item}\n') # final order
        nvert = len(item)
        npairs = nvert * (nvert-1) // 2
        g = Graph(nvert)
        gold_order = [i for i in range(nvert)]

        for i in range(nvert):
            for j in range(i+1, nvert):
                g.addEdge(item[i], item[j])

        stats.update_stats(nvert, npairs, item, gold_order, g)

    stats.print_stats()
