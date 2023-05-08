import torch
import numpy as np
import torch.nn as nn
import itertools
from torch.nn import functional as F

from .test_score import global_coherence_score

class Beam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, prob, prev_beam, f_done):
        pre_score = prob.new_tensor(prev_beam.scores)

        score = prob + pre_score.unsqueeze(-1).expand_as(prob)
        if score.numel() < self.beam_size:
            nbest_score, nbest_ix = score.view(-1).topk(score.numel(), largest=False)
        else:
            nbest_score, nbest_ix = score.view(-1).topk(self.beam_size, largest=False)

        beam_ix = nbest_ix // prob.size(1)
        token_ix = nbest_ix - beam_ix * prob.size(1)

        done_list, remain_list = [], []
        prev_candidates = prev_beam.candidates
        for b_score, b_ix, t_ix in itertools.zip_longest(nbest_score.tolist(), beam_ix.tolist(), token_ix.tolist()):
            candidate = prev_candidates[int(b_ix)] + [t_ix]

            if f_done(candidate):
                done_list.append([candidate, b_score])
            else:
                remain_list.append(b_ix)
                self.candidates.append(candidate)
                self.scores.append(b_score)
        return done_list, remain_list




class CoherenceBeam(object):
    def __init__(self, beam_size):
        self.beam_size = beam_size

        self.candidates = []
        self.scores = []

    def step(self, graph_model, glove_model, args, texts, prob, prev_beam, f_done):
        pre_score = prob.new_tensor(prev_beam.scores)

        # start from the second sentence
        if prev_beam.scores != [0]:
            new_text_id = []
            for cand in prev_beam.candidates:
                for i in range(prob.size(1)):
                    new_text_id.append(cand + [i])

            new_text = []
            for cand in new_text_id:
                new_text.append([texts[0][i] for i in cand])

            coherence_score = validateConfidence(graph_model, glove_model, new_text)
            coherence_score = torch.tensor(coherence_score).to(args.device)
            # print(f'prob: {prob}')
            # print(f'coherence_score: {coherence_score}')
            # print(f'pre_score: {pre_score}')
            assert len(coherence_score) == prob.size(1) * prob.size(0)
            coherence_score = coherence_score.unsqueeze(0).view(prob.size(0), prob.size(1))
            coherence_score = coherence_score.masked_fill(prob==-1e+9, -1e+9)
            # print(f'after coherence: {coherence_score}')
            score = args.alpha * prob + pre_score.unsqueeze(-1).expand_as(prob) + coherence_score
        else:
            # print(f'prob: {prob}')
            score = args.alpha * prob + pre_score.unsqueeze(-1).expand_as(prob)

        if score.numel() < self.beam_size:
            nbest_score, nbest_ix = score.view(-1).topk(score.numel(), largest=True)
        else:
            nbest_score, nbest_ix = score.view(-1).topk(self.beam_size, largest=True)

        beam_ix = nbest_ix // prob.size(1)
        token_ix = nbest_ix - beam_ix * prob.size(1)

        done_list, remain_list = [], []
        prev_candidates = prev_beam.candidates

        
        for b_score, b_ix, t_ix in itertools.zip_longest(nbest_score.tolist(), beam_ix.tolist(), token_ix.tolist()):
            candidate = prev_candidates[b_ix] + [t_ix]

            if f_done(candidate):
                done_list.append([candidate, b_score])
            else:
                remain_list.append(b_ix)
                self.candidates.append(candidate)
                self.scores.append(b_score)
        return done_list, remain_list


def validateConfidence(graph_model, texts):

    graph_score = global_coherence_score(graph_model, texts)
    score = [x for x in graph_score]

    return score
