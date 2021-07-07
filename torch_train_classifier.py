import os
import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.utils.data
import torch_network
import utils


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(
        torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


DATA_DIR = "ModelNet10"     # <- Set this path correctly

num_points_per_cloud = 4096     # <- you can modify this number as needed
train_pc, test_pc, train_labels, test_labels, class_ids = utils.create_point_cloud_dataset(
    DATA_DIR, num_points_per_cloud)

# once loaded save the numpy arrays to pickle files to use later
pickle.dump(train_pc, open("trainpc.pkl", "wb"))
pickle.dump(test_pc, open("testpc.pkl", "wb"))
pickle.dump(train_labels, open("trainlabels.pkl", "wb"))
pickle.dump(test_labels, open("testlabels.pkl", "wb"))
pickle.dump(class_ids, open("class_ids.pkl", "wb"))

# load the data from pickle files if already present
# train_pc = pickle.load(open("trainpc.pkl", "rb"))
# train_labels = pickle.load(open("trainlabels.pkl", "rb"))
# test_pc = pickle.load(open("testpc.pkl", "rb"))
# test_labels = pickle.load(open("testlabels.pkl", "rb"))
# class_ids = pickle.load(open("class_ids.pkl", "rb"))
