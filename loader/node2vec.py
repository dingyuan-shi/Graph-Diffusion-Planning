import numpy as np
from collections import deque
from gensim.models.word2vec import Word2Vec
import gensim
import pickle
from gensim.models.callbacks import CallbackAny2Vec
import random 
from tqdm import tqdm
import os
import pandas as pd
import networkx as nx


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''
    def __init__(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print('Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1
        
        
class Node2Vec:
    def __init__(self, graph: nx.Graph, embed_path, path_path, p=2, q=4) -> None:
        self.graph = graph
        self.p = p
        self.q = q
        self.embed_path = embed_path
        self.path_path = path_path
        self.neighbor_sampler = self._build_neighbors()
             
    def _build_alias_table(self, probs):
        prob_regular = [prob * len(probs) for prob in probs]
        larger = deque()
        smaller = deque()
        alias, alias_prob = [], []
        eps = 0.000000001
        for i in range(len(prob_regular)):
            if prob_regular[i] > 1. + eps:
                larger.append(i)
            elif prob_regular[i] < 1. - eps:
                smaller.append(i)
            else:
                alias.append((i, -1))
                alias_prob.append(1)
                
        while len(smaller) > 0:
            s = smaller.popleft()
            l = larger.popleft()
            alias.append((s, l))
            alias_prob.append(prob_regular[s])
            
            prob_regular[l] -= (1 - prob_regular[s])
            if prob_regular[l] > 1. + eps:
                larger.append(l)
            elif prob_regular[l] < 1. - eps:
                smaller.append(l)
            else:
                alias.append((l, -1))
                alias_prob.append(1)
        return alias, alias_prob

    def _build_neighbors(self):
        neighbor_sampler = dict()
        if os.path.exists(self.path_path):
            return pickle.load(open(self.path_path, "rb"))
        for prev in tqdm(self.graph):
            for node in self.graph:
                if node == prev: continue
                # from prev to node, sample from node
                candidates = [prev]
                prob_unorm = [1. / self.p]
                for adj in self.graph[node]:
                    if adj == prev: continue
                    candidates.append(adj)
                    prob_unorm.append(1. if adj in self.graph[prev] else 1. / self.q)
                all_probs = np.array(prob_unorm)
                all_probs /= np.sum(all_probs)        
                alias, alias_prob = self._build_alias_table(all_probs)
                neighbor_sampler[prev, node] = (alias, alias_prob, candidates)
        pickle.dump(neighbor_sampler, open(self.path_path, "wb"))
        return neighbor_sampler
        
    def _sample(self, alias, alias_prob, all_neighbors):
        p = np.random.randint(0, len(alias))
        if alias[p][1] == -1:
            return all_neighbors[alias[p][0]]
        r = np.random.uniform(0, 1)
        return all_neighbors[alias[p][0 if r < alias_prob[p] else 1]]
    
    def random_walk(self, start_node, length):
        walk = [start_node]
        while len(walk) < length:
            cur_node = walk[-1]
            if len(self.graph[cur_node]) == 0:
                break
            if len(walk) == 1:
                walk.append(random.choice(list(self.graph[cur_node].keys())))
            else:
                prev, cur = walk[-2], walk[-1]
                if prev == cur:
                    walk.append(random.choice(list(self.graph[cur].keys())))
                else:
                    walk.append(self._sample(*self.neighbor_sampler[prev, cur]))
        return walk

    def train(self, sample_num, sample_length, **kwargs):
        walks = []
        for node in tqdm(self.graph):
            for _ in range(sample_num):
                walk = self.random_walk(node, length=sample_length)
                walks.append(walk)

        print("Learning embedding vectors...")
        model = Word2Vec(sentences=walks, **kwargs)
        print("Learning embedding vectors done!")

        # get embeddings
        embeddings = dict()
        for node in self.graph:
            embeddings[node] = model.wv[node]
        pickle.dump(embeddings, open(self.embed_path, "wb"))


def get_node2vec(graph, embed_path, path_path, p=2, q=4):
    if not os.path.exists(embed_path):
        node2vec = Node2Vec(graph, embed_path, path_path, p, q)
        node_embed_size=100
        node2vec.train(sample_num=1000, sample_length=20, vector_size=node_embed_size, 
                        alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
                        sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=1, hs=0, 
                        negative=5, ns_exponent=0.75, cbow_mean=1, 
                        epochs=20, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, 
                        compute_loss=True, callbacks=[callback()], shrink_windows=True)
    return pickle.load(open(embed_path, "rb")) 
    