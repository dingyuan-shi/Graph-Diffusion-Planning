import torch
import networkx as nx
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
import transformers
from transformers.models.gpt2 import GPT2Model
from models_seq.seq_models import Restorer, Destroyer
from loader.dataset import TrajFastDataset
import pickle
from torch.nn.utils.rnn import pad_sequence


class Planner(nn.Module):
    
    def _calculate_unit_dir_vec(self, ya, xa, yb, xb):
        denom = ((yb - ya) ** 2 + (xb - xa) ** 2) ** 0.5
        if denom == 0.:
            return  (0., 0.)
        return ((yb - ya) / denom, (xb - xa) / denom)
    
    def __init__(self, G: nx.Graph, A: torch.Tensor, restorer: Restorer, destroyer: Destroyer, device: torch.device, x_emb_dim: int, pretrain_path=None):
        super().__init__()
        self.device = device
        # find max degree and build mask
        self.max_deg = A.long().sum(1).max()
        self.n_vertex = A.shape[0]
        self.mask = torch.zeros(self.n_vertex, self.max_deg + 1).long().to(self.device)
        self.mask[torch.arange(self.n_vertex), A.sum(1, keepdim=False).long()] = 1
        self.mask.cumsum_(dim=-1)
        self.mask = self.mask[:, :-1].bool()
        
        self.locations = torch.zeros([self.n_vertex, 2]).to(self.device)
        for k in range(self.n_vertex):
            self.locations[k, 0], self.locations[k, 1] = G.nodes[k]["lng"], G.nodes[k]["lat"]
        
        self.v_to_ord = dict()  # v : dict from v to ord
        self.ord_to_v = dict()  # v : [] list of vertices
        val, ind = A.long().topk(self.max_deg, dim=1)
        for i in range(self.n_vertex):
            valid_ind = ind[i][val[i] == 1].cpu().tolist()
            self.v_to_ord[i] = dict(zip(valid_ind, list(range(len(valid_ind)))))
            self.ord_to_v[i] = valid_ind
            
        # two vertex abs direction
        # (xb - xa)/ \sqrt{(y_b - y_a)^2 + (xb - xa)^2}, (yb - ya)/ \sqrt{(y_b - y_a)^2 + (xb - xa)^2}
        self.tv_dir = torch.zeros([self.n_vertex, self.n_vertex, 2]).to(self.device)
        self.adj_dir = torch.zeros(self.n_vertex, self.max_deg, 2).to(self.device)
        for k in range(self.n_vertex):
            xb_m_xa = self.locations[:, 0] - self.locations[k, 0]
            yb_m_ya = self.locations[:, 1] - self.locations[k, 1]
            denom = (xb_m_xa.square() + yb_m_ya.square()).sqrt()
            self.tv_dir[k, denom > 0, 0], self.tv_dir[k, denom > 0, 1] = (xb_m_xa / denom)[denom > 0], (yb_m_ya / denom)[denom > 0]
            # each vertex adjacent direction
            self.adj_dir[k, torch.arange(len(self.ord_to_v[k])), :] = self.tv_dir[k, self.ord_to_v[k], :]
        
        config = transformers.GPT2Config(vocab_size=1, n_embd=x_emb_dim, n_head=4, n_layer=6)
        self.transformer = GPT2Model(config).to(device)
        self.restorer = restorer
        self.destroyer = destroyer
        
        distance_dim = 50
        direction_dim = 50
        self.distance_mlp = nn.Linear(1, distance_dim).to(self.device)
        self.direction_mlp = nn.Linear(self.max_deg, direction_dim).to(self.device)
        hidden_dim = 100
        self.out_mlp = nn.Sequential(
            # hidden from gpt, distance, direction, destination
            nn.Linear(x_emb_dim + distance_dim + direction_dim + x_emb_dim, hidden_dim).to(self.device), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, int(0.5 * hidden_dim)).to(self.device), 
            nn.ReLU(), 
            nn.Linear(int(0.5 * hidden_dim), self.max_deg).to(self.device)
        )
        
        # load pretrained
        if pretrain_path is not None:
            node2vec = pickle.load(open(pretrain_path, "rb"))
            assert self.n_vertex == len(node2vec)
            if x_emb_dim != node2vec[0].shape[0]:
                print("Use pretrained embed dims")
            x_emb_dim = node2vec[0].shape[0]
            nodeemb = torch.zeros(self.n_vertex + 2, x_emb_dim)
            for k in node2vec:
                nodeemb[k] = torch.from_numpy(node2vec[k])
            self.x_embedding = nn.Embedding.from_pretrained(nodeemb, freeze=False).to(device)
        else:
            self.x_embedding = nn.Embedding(self.n_vertex + 2, x_emb_dim, padding_idx=self.n_vertex, device=device).to(self.device)
        
        
    def forward(self, xs):
        # xs: list of tensors
        lengths = [x.shape[0] for x in xs]
        destinations = torch.Tensor([x[-1] for x in xs]).long().to(self.device)
        xs_padded = pad_sequence(xs, batch_first=True, padding_value=0)
        xs_actions = [torch.Tensor([self.v_to_ord[a.item()][b.item()] for a, b in zip(x, x[1:])]).long().to(self.device) for x in xs]
        batch_size, horizon = xs_padded.shape
        xs_padded_emb = self.x_embedding(xs_padded)
        dest_emb = self.x_embedding(destinations)
        attention_mask = torch.ones_like(xs_padded).long()
        for k in range(batch_size):
            attention_mask[k, lengths[k]:] = 0
        
        transformer_outputs = self.transformer(
            inputs_embeds = xs_padded_emb, 
            attention_mask = attention_mask,
        )
        hidden = transformer_outputs['last_hidden_state']   # b h c = embed
        # get distances: b h 1
        distances = (self.locations[xs_padded] - self.locations[destinations].unsqueeze(1)).abs().sum(dim=-1, keepdim=True) * 100
        distances_feature = self.distance_mlp(distances) # b h dim
        # get directions: b h deg
        directions = (self.adj_dir[xs_padded] * self.tv_dir[xs_padded, destinations.unsqueeze(1)].unsqueeze(2)).sum(dim=-1, keepdim=False)
        # fill -1
        directions = torch.masked_fill(directions, self.mask[xs_padded], -1)
        directions_feature = self.direction_mlp(directions) # b h dim
        feed = torch.concat([hidden, distances_feature, directions_feature, dest_emb.unsqueeze(1).repeat(1, horizon, 1)], dim=-1)
        # distance_mlp, direction_mlp, hidden => out_mlp
        out_logits = self.out_mlp(feed)
        loss = sum([F.cross_entropy(out_logits[k][:lengths[k] - 1], xs_actions[k], reduction="mean") for k in range(batch_size)])
        return loss
    
    
    def plan(self, origs, dests, eval_nll=False):
        with torch.no_grad():
            window = 1
            # origs: list of vertices | tensor of vertices
            # dests list of vertices  | tensor of vertices
            if type(origs) is list:
                origs = torch.Tensor(origs).long().to(self.device)
            if type(dests) is list:
                dests = torch.Tensor(dests).long().to(self.device)
            origs_emb = self.x_embedding(origs)  # b c
            dests_emb = self.x_embedding(dests)  # b c
            batch_size, x_emb_dim = origs_emb.shape
            max_len = 100
            xs = torch.zeros([batch_size, max_len]).long().to(self.device)
            xs_emb = torch.zeros([batch_size, max_len, x_emb_dim]).to(self.device)
            xs[:, 0] = origs
            window_idx = window
            stop = torch.zeros([batch_size]).bool().to(self.device)
            actual_length = torch.ones([batch_size]).long().to(self.device) * max_len
            if eval_nll:
                nlls = torch.zeros([batch_size]).to(self.device)
            for i in range(1, max_len):
                prefix = xs[:, :i]
                prefix_emb = self.x_embedding(prefix)
                # proposal from transformer            
                transformer_outputs = self.transformer(
                    inputs_embeds = prefix_emb, 
                )
                hidden = transformer_outputs['last_hidden_state']   # b h c = x_embed
                hidden = hidden[:, -1, :]  # only need the last one
                # get distances: b 2
                distances = (self.locations[prefix[:, -1]] - self.locations[dests]).square().sum(dim=-1, keepdim=True).sqrt() * 100
                distances_feature = self.distance_mlp(distances) # b dim
                # get directions: b deg
                directions = (self.adj_dir[prefix[:, -1]] * self.tv_dir[prefix[:, -1], dests].unsqueeze(1)).sum(dim=-1, keepdim=False)
                directions = torch.masked_fill(directions, self.mask[prefix[:, -1]], -1)
                directions_feature = self.direction_mlp(directions) # b dim
                feed = torch.concat([hidden, distances_feature, directions_feature, dests_emb], dim=-1)
                out_logits_gpt = self.out_mlp(feed)
                out_logits_gpt = torch.masked_fill(out_logits_gpt, self.mask[prefix[:, -1]], value=-1e20)
                gpt_probs = torch.softmax(out_logits_gpt, dim=-1)
                # proposal from diffusion
                # concat and window
                if window_idx == window:
                    window = min(window * 2, max_len)  # exponentially increasing
                    ts = torch.Tensor([self.destroyer.max_T]).long().repeat(batch_size).to(self.device)
                    prefix_diffused = self.destroyer.diffusion(prefix, ts, ret_distr=False)
                    prefix_diffused = pad_sequence(prefix_diffused, batch_first=True, padding_value=0)
                    pure_random = torch.randint(0, self.n_vertex, [batch_size, window]).to(self.device)
                    real_fake = torch.concat([prefix_diffused, pure_random], dim=-1)
                    lengths = torch.Tensor([i - 1 + window]).long().repeat(batch_size).to(self.device)
                    _, x0_pred_probs = self.restorer.sample_with_len(lengths, xt=real_fake, ret_distr=True)
                    proposal = x0_pred_probs[:, -window:]  # b window v
                    window_idx = 0
                # syntheis
                cur_proposal = proposal[:, window_idx, :]
                window_idx += 1
                diff_probs_list = [cur_proposal[k, self.ord_to_v[prefix[k, -1].item()]] for k in range(batch_size)]
                for k in range(batch_size):
                    diff_probs_list[k][diff_probs_list[k] < 1e-3] = 1e-3
                diff_probs_padded = pad_sequence(diff_probs_list, batch_first=True, padding_value=0)
                diff_probs = torch.zeros_like(gpt_probs).to(self.device)
                diff_probs[:, :diff_probs_padded.shape[1]] = diff_probs_padded
                syntheised_probs = diff_probs * gpt_probs
                syntheised_probs = syntheised_probs / syntheised_probs.sum(1, keepdim=True)
                # actions = torch.multinomial(syntheised_probs, 1).squeeze()
                actions = torch.argmax(syntheised_probs, 1)
                # action to vertex
                xs[:, i] = torch.Tensor([self.ord_to_v[prefix[k, -1].item()][actions[k]] for k in range(batch_size)]).long().to(self.device)
                if eval_nll:
                    nlls[~stop] -= (syntheised_probs[~stop, actions[~stop]] + 0.0001).log()
                actual_length[xs[:, i] == dests] = i + 1
                stop = stop | (xs[:, i] == dests)
                if stop.all():
                    break
                xs_list = [xs[k, :actual_length[k]].cpu().tolist() for k in range(batch_size)]
                xs_list_refined = self.refine(xs_list, dests.cpu().tolist())
            if eval_nll:
                return xs_list_refined, nlls
            return xs_list_refined 
    
    def refine(self, paths, dests):
        # two things: 1) cut the recursive
        refined_paths = []
        for k, path in enumerate(paths):
            # 1) if one step close, directly cut
            destination = dests[k]
            for i, v in enumerate(path):
                if destination in self.v_to_ord[v]:
                    cutted_path = path[:i] + [destination]
                    break
            else:
                cutted_path = path
            # 2) cut the recursive
            showup = set()
            points_filtered = []
            for _, v in enumerate(cutted_path):
                if v not in showup:
                    showup.add(v)
                    points_filtered.append(v)
                else:
                    while points_filtered[-1] != v:
                        showup.discard(points_filtered[-1])
                        points_filtered.pop()
            refined_paths.append(points_filtered)
        return refined_paths