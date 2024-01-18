import torch
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import cycle
from os.path import join

from .planner import Planner
from loader.dataset import TrajFastDataset


class Trainer:
    def __init__(self, model: Planner, dataset: TrajFastDataset, device: torch.device, model_path: str, test_dataset=None):
        self.model = model 
        self.train_dataset = dataset
        self.device = device
        self.model_path = model_path
        if test_dataset is None:
            train_num = int(0.8 * len(self.train_dataset))
            self.train_dataset, self.test_dataset = random_split(self.train_dataset, [train_num , len(self.train_dataset) - train_num])
        
    def train(self, n_epoch, batch_size, lr):
        optimizer = torch.optim.Adam(self.model.parameters(), lr)
        # split train test
        trainloader = DataLoader(self.train_dataset, batch_size, 
                                collate_fn=lambda data: [torch.Tensor(each).long().to(self.device) for each in data])
        testloader = DataLoader(self.test_dataset, batch_size, 
                                collate_fn=lambda data: [torch.Tensor(each).long().to(self.device) for each in data])
        self.model.train()
        iter, train_loss_avg = 0, 0
        try:
            for epoch in range(n_epoch):
                for xs in trainloader:
                    loss = self.model(xs)
                    train_loss_avg += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # TODO: clip norm
                    optimizer.step()
                    iter += 1
                    if iter % 100 == 0 or iter == 1:
                        denom = 1 if iter == 1 else 100
                        # eval test
                        test_loss = next(self.eval_test(testloader))
                        print(f"e: {epoch}, i: {iter}, train loss: {train_loss_avg / denom: .4f}, test loss: {test_loss: .4f}")
                        train_loss_avg = 0.
        except KeyboardInterrupt as E:
            print("Training interruptted, begin saving...")
            self.model.eval()
            model_name = f"tmp_iter_{iter}.pth"
        # save
        self.model.eval()
        model_name = f"finished_{iter}.pth"
        torch.save(self.model, join(self.model_path, model_name))
        print("save finished!")
        
        
            
    def eval_test(self, test_loader):
        with torch.no_grad():
            for txs in cycle(test_loader):
                test_loss = self.model(txs)
                yield test_loss.item()

