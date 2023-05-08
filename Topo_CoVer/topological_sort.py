# Credits: The code for this file is based on https://github.com/shrimai/Topological-Sort-for-Sentence-Ordering

from collections import defaultdict 


class Graph: 
    '''
    The code for this class is based on geeksforgeeks.com
    '''
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices 
        self.visited = [False]*vertices
        self.recStack = [False]*vertices
  
    def addEdge(self, u, v): 
        self.graph[v].append(u) # latter to former
        # self.graph[u].append([v, w]) # former to latter

    def printGraph(self):
        print(f'graph: {self.graph}')

    def getCyclePos(self):
        cycle_pos = [i for i in range(len(self.recStack)) if self.recStack[i] == True]
        for node in cycle_pos:
            self.recStack[node] = False
            self.visited[node] = True
        return cycle_pos


    def findFirstSentence(self, verified=True):
        for i in range(self.V):
            if len(self.graph[i]) == 0 and self.visited[i] == False:
                if verified:
                    self.visited[i] = True
                return i

    def removeEdge(self, v):
        for i in range(self.V):
            if v in self.graph[i]:
                self.graph[i].remove(v)
        
    def isCyclicUtil(self, v, visited): 
  
        visited[v] = True
        self.recStack[v] = True
  
        for neighbour in self.graph[v]:
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited) == True: 
                    return True
            elif self.recStack[neighbour] == True: 
                self.graph[v].remove(neighbour)
                return True
  
        self.recStack[v] = False
        return False
  
    def isCyclic(self): 
        visited = [False] * self.V 

        for node in range(self.V):
            if self.visited[node] == False and visited[node] == False: 
                if self.isCyclicUtil(node, visited) == True: 
                    return True
        return False

class Stats(object):
    
    def __init__(self):
        self.n_samp = 0
        self.n_sent = 0
        self.n_pair = 0
        self.corr_samp = 0
        self.corr_sent = 0
        self.corr_pair = 0
        self.lcs_seq = 0
        self.tau = 0
        self.dist_window = [1, 2, 3]
        self.min_dist = [0]*len(self.dist_window)
        self.fm = 0
        self.lm = 0
        
    
    def kendall_tau(self, porder, gorder):
        '''
        It calculates the number of inversions required by the predicted 
        order to reach the correct order.
        '''
        pred_pairs, gold_pairs = [], []
        for i in range(len(porder)):
            for j in range(i+1, len(porder)):
                pred_pairs.append((porder[i], porder[j]))
                gold_pairs.append((gorder[i], gorder[j]))
        common = len(set(pred_pairs).intersection(set(gold_pairs)))
        uncommon = len(gold_pairs) - common
        tau = 1 - (2*(uncommon/len(gold_pairs)))

        return tau
    
    def min_dist_metric(self, porder, gorder):
        '''
        It calculates the displacement of sentences within a given window.
        '''
        count = [0]*len(self.dist_window)
        for i in range(len(porder)):
            pidx = i
            pval = porder[i]
            gidx = gorder.index(pval)
            for w, window in enumerate(self.dist_window):
                if abs(pidx-gidx) <= window:
                    count[w] += 1
        return count
    
    def lcs(self, X , Y): 
        m = len(X) 
        n = len(Y) 

        L = [[None]*(n+1) for i in range(m+1)] 

        for i in range(m+1): 
            for j in range(n+1): 
                if i == 0 or j == 0 : 
                    L[i][j] = 0
                elif X[i-1] == Y[j-1]: 
                    L[i][j] = L[i-1][j-1]+1
                else: 
                    L[i][j] = max(L[i-1][j] , L[i][j-1]) 

        return L[m][n] 
    
    def sample_match(self, order, gold_order):
        '''
        It calculates the percentage of samples for which the entire 
        sequence was correctly predicted. (PMR)
        '''
        return order == gold_order
    
    
    def first_match(self, order, gold_order):
        '''
        It calculates the percentage of samples for which the first sentence 
        was correctly predicted. (PMR)
        '''
        return order[0] == gold_order[0]
    
    def last_match(self, order, gold_order):
        '''
        It calculates the percentage of samples for which the first sentence 
        was correctly predicted. (PMR)
        '''
        return order[-1] == gold_order[-1]
    
    def sentence_match(self, order, gold_order):
        '''
        It measures the percentage of sentences for which their absolute 
        position was correctly predicted. (Acc)
        '''
        return sum([1 for x in range(len(order)) if order[x] == gold_order[x]])
    
    def update_stats(self, nvert, npairs, order, gold_order, g):
        self.n_samp += 1
        self.n_sent += nvert
        self.n_pair += npairs
        
        if self.sample_match(order, gold_order):
            self.corr_samp += 1
        if self.first_match(order, gold_order):
            self.fm += 1
        if self.last_match(order, gold_order):
            self.lm += 1
        self.corr_sent += self.sentence_match(order, gold_order)
        self.lcs_seq += self.lcs(order, gold_order)
        self.tau += self.kendall_tau(order, gold_order)
        window_counts = self.min_dist_metric(order, gold_order)
        for w, wc in enumerate(window_counts):
            self.min_dist[w] += wc
        
    def print_stats(self):
        print("Perfect Match: " + str(self.corr_samp*100/self.n_samp))
        print("First Sentence Match: " + str(self.fm*100/self.n_samp))
        print("Last Sentence Match: " + str(self.lm*100/self.n_samp))
        print("Sentence Accuracy: " + str(self.corr_sent*100/self.n_sent))
        print("LCS: " + str(self.lcs_seq*100/self.n_sent))
        print("Kendall Tau Ratio: " + str(self.tau/self.n_samp))
        for w, window in enumerate(self.dist_window):
            print("Min Dist Metric for window " + str(window) + ": " + \
                                    str(self.min_dist[w]*100/self.n_sent))
            
    def metric(self):
        PMR = str(round(self.corr_samp*100/self.n_samp, 4))
        tau = str(round(self.tau/self.n_samp, 4))
        return {"PMR": PMR, "tau": tau}
