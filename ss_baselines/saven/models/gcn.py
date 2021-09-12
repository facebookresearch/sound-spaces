# Gyan Tatiya

import pickle

import scipy.sparse as sp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj):

    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()

        # get and normalize adjacency matrix.
        adjmat_path = r"data/glove_data/adjmat.bin"
        bin_file = open(adjmat_path, "rb")
        A_raw = pickle.load(bin_file)
        bin_file.close()
        # print("A_raw: ", A_raw.shape, A_raw)
        A = normalize_adj(A_raw).tocsr().toarray()
        self.A = torch.nn.Parameter(torch.Tensor(A))
        # print("self.A: ", self.A.shape)
        # print("self.A: ", self.A)

        embeddings_path = r"data/glove_data/glove_embeddings_300d.bin"

        bin_file = open(embeddings_path, "rb")
        objects_vector = pickle.load(bin_file)
        regions_vector = pickle.load(bin_file)
        bin_file.close()

        objects = list(sorted(objects_vector.keys()))
        regions = list(sorted(regions_vector.keys()))

        self.n = len(objects) + len(regions)

        # print("objects: ", len(objects), objects)
        # print("regions: ", len(regions), regions)
        # print("words: ", len(words), words)

        all_glove = torch.zeros(self.n, 300)
        i = 0
        for obj in objects:
            all_glove[i, :] = torch.Tensor(objects_vector[obj])
            i += 1
        for reg in regions:
            all_glove[i, :] = torch.Tensor(regions_vector[reg])
            i += 1
        # print("all_glove: ", all_glove.shape)
        # print("all_glove ", all_glove)

        self.all_glove = nn.Parameter(all_glove)
        self.all_glove.requires_grad = False

        self.get_word_embed = nn.Linear(300, self.n)

        self.W0 = nn.Linear(self.n*2, 1024, bias=False)
        self.W1 = nn.Linear(1024, 1024, bias=False)
        self.W2 = nn.Linear(1024, 1, bias=False)

        self.final_mapping = nn.Linear(self.n, 256-2)

    def forward(self, class_embed):

        class_embed = class_embed.reshape(1, -1)
        # print("class_embed: ", class_embed.shape)
        word_embed = self.get_word_embed(self.all_glove.detach())
        # print("word_embed: ", word_embed.shape)
        x = torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1)
        # print("torch.cat((class_embed.repeat(self.n, 1), word_embed), dim=1): ", x.shape)
        x = torch.mm(self.A, x)
        # print("torch.mm(self.A, x): ", x.shape)
        x = F.relu(self.W0(x))
        # print("F.relu(self.W0(x)): ", x.shape)
        x = torch.mm(self.A, x)
        # print("torch.mm(self.A, x): ", x.shape)
        x = F.relu(self.W1(x))
        # print("F.relu(self.W1(x)): ", x.shape)
        x = torch.mm(self.A, x)
        # print("torch.mm(self.A, x): ", x.shape)
        x = F.relu(self.W2(x))
        # print("F.relu(self.W2(x)): ", x.shape)
        x = x.view(1, self.n)
        # print("x.view(1, self.n): ", x.shape)
        x = self.final_mapping(x)
        # print("self.final_mapping(x): ", x.shape)

        return x
