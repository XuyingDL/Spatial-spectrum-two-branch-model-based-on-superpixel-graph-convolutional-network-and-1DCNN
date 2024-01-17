'''Predicting'''
from scipy.io import loadmat, savemat
import numpy as np
import argparse
import configparser
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from skimage.segmentation import slic, mark_boundaries
from sklearn.preprocessing import scale
import os
from PIL import Image
from utils import get_graph_list, get_edge_index
import math
from module import SubGcnFeature, GraphNet, CNN
from Trainer1 import JointTrainer
torch.backends.cudnn.enabled = False
import matplotlib.pyplot as plt
import pandas as pd
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAIN THE OVERALL')
    parser.add_argument('--name', type=str, default='hubei',
                        help='DATASET NAME')
    parser.add_argument('--block', type=int, default=5,
                        help='BLOCK SIZE')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--comp', type=int, default=10,
                        help='COMPACTNESS')
    parser.add_argument('--batchsz', type=int, default=64,
                        help='BATCH SIZE')
    parser.add_argument('--run', type=int, default=1,
                        help='EXPERIMENT AMOUNT')
    parser.add_argument('--spc', type=int, default=400,
                        help='SAMPLE per CLASS')
    parser.add_argument('--hsz', type=int, default=128,
                        help='HIDDEN SIZE')
    parser.add_argument('--hsz1d', type=int, default=128,
                        help='HIDDEN SIZE 1DCNN')
    parser.add_argument('--band', type=int, default=39,
                        help='BAND')
    parser.add_argument('--label', type=int, default=2,
                        help='LABEL')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='LEARNING RATE')
    parser.add_argument('--wd', type=float, default=0.,
                        help='WEIGHT DECAY')
    arg = parser.parse_args()
    config = configparser.ConfigParser()
    config.read('dataInfo.ini')

    # Data processing
    # Reading hyperspectral image
    data_path = 'data/{0}/{0}.mat'.format(arg.name)
    m = loadmat(data_path)
    data = m['{0}'.format(arg.name)]
    gt_path = 'data/{0}/{0}_gt.mat'.format(arg.name)
    m = loadmat(gt_path)
    gt = m['{0}_gt'.format(arg.name)]
    # Normalizing data
    h, w, c = data.shape
    data = data.reshape((h * w, c))
    data = data.astype(np.float)
    data_normalization = scale(data).reshape((h, w, c))


    # Superpixel segmentation
    seg_root = 'data/rgb'
    seg_path = os.path.join(seg_root, '{}_seg_{}.npy'.format(arg.name, arg.block))
    if os.path.exists(seg_path):
        seg = np.load(seg_path)
    else:
        rgb_path = os.path.join(seg_root, '{}_rgb.jpg'.format(arg.name))
        img = Image.open(rgb_path)
        img_array = np.array(img)
        # The number of superpixel
        n_superpixel = int(math.ceil((h * w) / arg.block))
        seg = slic(img_array, n_superpixel, arg.comp)
        # Saving
        np.save(seg_path, seg)

    # Constructing graphs
    graph_path = 'data/{}/{}_graph.pkl'.format(arg.name, arg.block)
    if os.path.exists(graph_path):
        graph_list = torch.load(graph_path)
    else:
        graph_list = get_graph_list(data_normalization, seg)
        torch.save(graph_list, graph_path)
    subGraph = Batch.from_data_list(graph_list)

    # Constructing full graphs
    full_edge_index_path = 'data/{}/{}_edge_index.npy'.format(arg.name, arg.block)
    if os.path.exists(full_edge_index_path):
        edge_index = np.load(full_edge_index_path)
    else:
        edge_index, _ = get_edge_index(seg)
        np.save(full_edge_index_path,
                edge_index if isinstance(edge_index, np.ndarray) else edge_index.cpu().numpy())
    fullGraph = Data(None,
                    edge_index=torch.from_numpy(edge_index) if isinstance(edge_index, np.ndarray) else edge_index,
                    seg=torch.from_numpy(seg) if isinstance(seg, np.ndarray) else seg)

    test = np.array(fullGraph.edge_index)
    test = test - 1
    fullGraph.edge_index = torch.tensor(test)


    gcn1 = SubGcnFeature(arg.band, arg.hsz, arg.label)
    gcn2 = GraphNet(arg.hsz, arg.hsz, arg.label)
    cnn = CNN(1, arg.hsz1d, arg.label)

    device = torch.device('cuda:{}'.format(arg.gpu)) if arg.gpu != -1 else torch.device('cpu')

    for r in range(arg.run):
        # Loading pretraining parameters
        gcn1.load_state_dict(
                torch.load(f"models/{arg.name}/400/{arg.block}_overall_skip_2_SGConv_l1_clip2/1DCNN+SGCN1/intNet_best_{arg.spc}_{r}.pkl"))
        gcn2.load_state_dict(
                torch.load(f"models/{arg.name}/400/{arg.block}_overall_skip_2_SGConv_l1_clip2/1DCNN+SGCN1/extNet_best_{arg.spc}_{r}.pkl"))
        cnn.load_state_dict(
            torch.load(f"models/{arg.name}/400/{arg.block}_overall_skip_2_SGConv_l1_clip2/1DCNN+SGCN1/1DCNN_best_{arg.spc}_{r}.pkl"))
        trainer = JointTrainer([gcn1, gcn2, cnn])

        logitsint, logits, logitscnn = trainer.predict(subGraph, fullGraph, device)
        seg=seg-1
        seg_torch = torch.from_numpy(seg).reshape(-1,).squeeze()
        logitsout = logits[seg_torch]
        pred = (0.5*logitsout + 0.5*logitsint)*0.5+0.5*logitscnn
        prob = torch.softmax(pred, dim=1)
        prob = prob[:, 1]
        result = prob.reshape(-1, 1)


        save_root = 'prediction/1DCNN+SGCN1'
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, '1DCNN+SGCN1.csv')
        pd.DataFrame(result.cpu().numpy()).to_csv(save_path, sep=',', index=False, header='true')
    print('*'*5 + 'FINISH' + '*'*5)

