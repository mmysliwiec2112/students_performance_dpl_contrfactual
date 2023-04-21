import torch
from deepproblog.query import Query
from deepproblog.dataset import Dataset
from problog.logic import Term, Constant
import os
import torchvision.transforms as transforms

path = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


class Performance(Dataset):
    """
    A class that inherits after the Dataset class from deepproblog library, that converts records from the
    students scores dataset to queries in dpl.
    Dataset has fields: gender, ethnicity, parental_education, lunch, test_completed, score3, score2, score3
    For now, I am trying to make predictions only on one of the scores, then expand to the rest.

    I'm not sure if it is needed in the whole project, because I found a way to import a tensor by adding this line:
    model.add_tensor_source("train", {'train': train_set})
    Still, I left it in the project for now, altough it is not used.

    I also have some doubts about the way the predictions are made, I am mainly looking at the coins example
    from the examples in dpl repository

    param: subset - name of the subset
    dataset: torch.Tensor - a tensor imported and converted to query-interpretable form
    """
    def __init__(
            self, subset, dataset: torch.Tensor
    ):
        super().__init__()
        self.data = []
        self.subset = subset
        for line in dataset:
            gender, ethnicity, parental_education, lunch, test_completed, s1, s2, s3 = line.tolist()

            self.data.append((gender, ethnicity, parental_education, lunch, test_completed, s1, outcome))

    def to_query(self, i):
        gender, ethnicity, parental_education, lunch, test_completed, s1, outcome = self.data[i]
        sub = {Term("a"): Term("tensor", Term(self.subset, Constant(i)))}
        return Query(Term("game", Term("a"), Term(outcome)), sub)

    def __len__(self):
        return len(self.data)

