from sklearn.mixture import GaussianMixture
from math import exp
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, random_split
from models_seq.eps_models import EPSM
from loader.dataset import TrajFastDataset
from models_seq.seq_models import Destroyer, Restorer
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
from os.path import join
from utils.coors import wgs84_to_gcj02


class Trainer:
    def __init__(self, model: nn.Module, dataset, model_path):
        self.model = model 
        self.device = model.device
        self.dataset = dataset
        self.model_path = model_path
        
    def train(self, n_epoch, batch_size, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # split train test
        train_num = int(0.8 * len(self.dataset))
        train_dataset, test_dataset = random_split(self.dataset, [train_num , len(self.dataset) - train_num])
        
        trainloader = DataLoader(train_dataset, batch_size, 
                                    collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        testloader = DataLoader(test_dataset, batch_size, 
                                collate_fn=lambda data: [torch.Tensor(each).to(self.device) for each in data])
        self.model.train()
        iter, train_loss_avg = 0, 0
        kl_loss_avg, ce_loss_avg, con_loss_avg = 0, 0, 0
        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    kl_loss, ce_loss, con_loss = self.model(xs)
                    if ce_loss.item() < 60:
                        loss =  kl_loss
                    else:
                        loss = kl_loss + ce_loss + con_loss
                        # loss = kl_loss + ce_loss
                    train_loss_avg += loss.item()
                    kl_loss_avg += kl_loss.item()
                    ce_loss_avg += ce_loss.item()
                    con_loss_avg += con_loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: clip norm
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0 or iter == 1:
                        # eval test
                        denom = 1 if iter == 1 else 100
                        test_kl, test_ce, test_con = next(self.eval_test(testloader))
                        test_loss = test_kl + test_ce + test_con
                        print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, (kl: {kl_loss_avg / denom: .4f}, ce: {ce_loss_avg / denom: .4f}, co: {con_loss_avg / denom: .4f}), test loss: {test_loss: .4f}, (kl: {test_kl: .4f}, ce: {test_ce: .4f}, co: {test_con: .4f})")
                        train_loss_avg, kl_loss_avg, ce_loss_avg, con_loss_avg = 0., 0., 0., 0.
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        # model_name = f"finished_{iter}.pth"
        # torch.save(self.model, join(self.model_path, model_name))
        # print("save finished!")
        
    def train_gmm(self, gmm_samples, n_comp):
        gmm = GaussianMixture(n_components=n_comp, covariance_type="tied")
        gmm_samples = min(len(self.dataset), gmm_samples)
        lenghts = np.array([len(self.dataset[k]) for k in range(gmm_samples)]).reshape(-1, 1)
        gmm.fit(lenghts)
        self.model.gmm = gmm  
        
            
    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                kl_loss, ce_loss, test_con = self.model(txs)
                yield (kl_loss.item(), ce_loss.item(), test_con.item())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    max_T = 100
    dataset = TrajFastDataset("chengdu", ["20161101"], "./sets_data", device, is_pretrain=True)
    betas = torch.linspace(0.0001, 10, max_T)
    # old beta: 0.01, 15, 50
    
    destroyer = Destroyer(dataset.A, betas, max_T, device)
    eps_model = EPSM(dataset.n_vertex, x_emb_dim=50, hidden_dim=20, dims=[100, 120, 200], device=device, pretrain_path="./sets_data/chengdu_node2vec.pkl")
    restorer = Restorer(eps_model, destroyer, device)
    
    trainer = Trainer(restorer, dataset, device, "./sets_model")
    trainer.train_gmm(gmm_samples=50000, n_comp=5)
    trainer.train(n_epoch=50, batch_size=16, lr=0.0005)
    
    restorer.eval()
    paths = restorer.sample_wo_len(100)
    
    multiple_locs = []
    for path in paths:
        locs = [[wgs84_to_gcj02(dataset.G.nodes[v]["lng"], dataset.G.nodes[v]["lat"])[1], 
                 wgs84_to_gcj02(dataset.G.nodes[v]["lng"], dataset.G.nodes[v]["lat"])[0]] 
                for v in path]
        multiple_locs.append(locs)
        print(locs)
    
    