import torch
from torch_geometric.data import Data, Batch
from torch.optim import optimizer as optimizer_
from torch_geometric.utils import accuracy
from torch_geometric.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
import numpy as np
import time


class JointTrainer(object):
    r'''Joint trainer'''
    def __init__(self, models: list):
        super().__init__()
        self.models = models

    def train(self, subGraph: Batch, fullGraph: Data, optimizer, criterion, device, monitor = None, is_l1=False, is_clip=False):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        CNN = DataParallel(self.models[2])

        intNet.train()
        extNet.train()
        CNN.train()

        intNet.to(device)
        extNet.to(device)
        CNN.to(device)
        criterion.to(device)

        logitscnn = CNN(subGraph.to_data_list())
        indices = torch.nonzero(subGraph.tr_gt.reshape(-1, )).squeeze()

        logitscnn = logitscnn[indices]


        fe, logitsint = intNet(subGraph.to_data_list())
        indices = torch.nonzero(subGraph.tr_gt.reshape(-1,)).squeeze()
        logitsint = logitsint[indices]


        # External graph features
        fullGraph.x = fe
        fullGraph = fullGraph.to(device)
        logits = extNet(fullGraph)
        indices = torch.nonzero(fullGraph.tr_gt, as_tuple=True)
        y = fullGraph.tr_gt[indices].to(device) - 1
        node_number = fullGraph.seg[indices]
        pixel_logits = logits[node_number]

        logits_GIG = 0.5 * pixel_logits + 0.5 * logitsint
        logits_all = 0.5*logits_GIG + 0.5*logitscnn

        logits = torch.softmax(logits_all,dim=1)
        pred = torch.argmax(logits, dim=-1)

        loss = criterion(logits, y)
        # l1 norm
        if is_l1:
            l1 = 0
            for p in intNet.parameters():
                l1 += p.norm(1)
            loss += 1e-4 * l1
        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        # Clipping gradient
        if is_clip:
            # External gradient
            clip_grad_norm_(extNet.parameters(), max_norm=2., norm_type=2)
            # Internal gradient
            clip_grad_norm_(intNet.parameters(), max_norm=3., norm_type=2)
        optimizer.step()

        if monitor is not None:
            monitor.add([intNet.parameters(), extNet.parameters()], ord=2)
        return loss.item(), accuracy(pred, y)

    def evaluate(self, subGraph, fullGraph, criterion, device):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        CNN = DataParallel(self.models[2])
        intNet.eval()
        extNet.eval()
        CNN.eval()
        intNet.to(device)
        extNet.to(device)
        CNN.to(device)
        criterion.to(device)
        with torch.no_grad():
            # subGraph = subGraph.to(device)
            logitscnn=CNN(subGraph.to_data_list())
            fe, logitsint = intNet(subGraph.to_data_list())
            indices1 = torch.nonzero(subGraph.te_gt.reshape(-1, )).squeeze()
            logitsint = logitsint[indices1]
            logitscnn = logitscnn[indices1]

            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            logits = extNet(fullGraph)

            indices = torch.nonzero(fullGraph.te_gt, as_tuple=True)
            y = fullGraph.te_gt[indices].to(device) - 1
            node_number = fullGraph.seg[indices]
            pixel_logits = logits[node_number]
            logits_all = (0.5 * pixel_logits + 0.5 * logitsint)*0.5 + 0.5*logitscnn
            logits = torch.softmax(logits_all, dim=1)
            pred = torch.argmax(logits, dim=-1)

            loss = criterion(logits, y)
        return loss.item(), accuracy(pred, y)

    # Getting prediction results
    def predict(self, subGraph, fullGraph, device: torch.device):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        CNN = DataParallel(self.models[2])
        intNet.eval()
        extNet.eval()
        CNN.eval()
        intNet.to(device)
        extNet.to(device)
        CNN.to(device)
        with torch.no_grad():

            logitscnn=CNN(subGraph.to_data_list())
            # Internal graph features
            fe, logitsint = intNet(subGraph.to_data_list())
            # External graph features
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            logits = extNet(fullGraph)
        pred = torch.softmax(logits, dim=1)

        pred = pred[:,1]

        return logitsint, logits,logitscnn

    # Getting hidden features
    def getHiddenFeature(self, subGraph, fullGraph, device, gt = None, seg = None):
        intNet = DataParallel(self.models[0])
        extNet = self.models[1]
        CNN = DataParallel(self.models[2])
        intNet.eval()
        extNet.eval()
        CNN.eval()
        intNet.to(device)
        extNet.to(device)
        CNN.to(device)
        with torch.no_grad():
            fe, logitsint = intNet(subGraph.to_data_list())
            fullGraph.x = fe
            fullGraph = fullGraph.to(device)
            fe = extNet(fullGraph)
        if gt is not None and seg is not None:
            indices = torch.nonzero(gt, as_tuple=True)
            gt = gt[indices] - 1
            node_number = seg[indices].to(device)
            fe = fe[node_number]
            return fe.cpu(), gt
        else:
            return fe.cpu()

    def get_parameters(self):
        return self.models[0].parameters(), self.models[1].parameters(), self.models[2].parameters()

    def save(self, paths):
        torch.save(self.models[0].cpu().state_dict(), paths[0])
        torch.save(self.models[1].cpu().state_dict(), paths[1])
        torch.save(self.models[2].cpu().state_dict(), paths[2])



