from typing import Mapping

import pandas as pd
import numpy as np
import zipfile
import os
import torch

from students_performance_dpl_contrfactual.network import MNetwork
from students_performance_dpl_contrfactual.dataset import Performance

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.evaluate import get_confusion_matrix


"""
Main program, for now the main goal is to get the model to give predictions with using the dpl model
"""

if not os.path.exists('./dataset/exams.csv'):
    with zipfile.ZipFile('./dataset.zip', 'r') as data_file:
        data_file.extractall('./dataset')

dataset = pd.read_csv("./dataset/exams.csv")


# Encoding parameters of the dataset
encoder = LabelEncoder()
for col in dataset.columns[:-3]:
    encoder.fit(dataset[col])
    dataset[col] = encoder.transform(dataset[col])

# Normalizing labels of the dataset
for col in dataset.columns[-3:]:
    dataset[col] = pd.Series(map(lambda x: 0 if x < 50 else 1, dataset[col].tolist()))

# creating train and test sets with dpl
train_set, test_set = train_test_split(dataset, test_size=0.33)
train_set = torch.tensor(train_set.values, dtype=torch.float32)
test_set = torch.tensor(test_set.values, dtype=torch.float32)

# transforming to the dpl friendly model
train_dataset = Performance("train", train_set)
test_dataset = Performance("test", test_set)
loader = DataLoader(train_dataset, 2)

# creation of neural network with dpl
mnet = MNetwork()
net = Network(mnet, "student_score", batching=True)
net.optimizer = torch.optim.Adam(mnet.parameters(), lr=1e-3)

# creation of the dpl model
model = Model('./model.pl', [net])
model.set_engine(ExactEngine(model), cache=True)
model.add_tensor_source("train", {'train': train_set})
model.add_tensor_source("test", {'test': test_set})

train_model(model, loader, 1, log_iter=100, profile=0)
print(get_confusion_matrix(model, test_dataset, verbose=1))
