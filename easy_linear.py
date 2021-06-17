import torch
import random
import numpy as np
import torch.utils.data as Data
import torch.nn as nn


num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
dataset = Data.TensorDataset(features, labels)

data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break


class LinearNet(nn.Moudle):
    def __init__(self, n_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    def forward(self,x):
        y = self.linear(x)
        return y

net = LinearNet(num_inputs)
print(net)