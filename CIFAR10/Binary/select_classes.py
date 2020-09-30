import numpy as np
import torchvision
import torch
from torch.utils.data import Dataset

def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class

def get_dataloaders(transform_train, transform_test, batch_size_train=128, batch_size_test=512):
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=None)
    classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,
                 'truck': 9}

    # Separating trainset/testset data/label
    x_train = trainset.data
    x_test = testset.data
    y_train = trainset.targets
    y_test = testset.targets

    # Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
    car_truck_trainset = \
        DatasetMaker(
            [get_class_i(x_train, y_train, classDict['truck']), get_class_i(x_train, y_train, classDict['car'])],
            transform_train
        )
    car_truck_testset = \
        DatasetMaker(
            [get_class_i(x_test, y_test, classDict['truck']), get_class_i(x_test, y_test, classDict['car'])],
            transform_test
        )
    trainloader = torch.utils.data.DataLoader(
        car_truck_trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        car_truck_testset, batch_size=batch_size_test, shuffle=True, num_workers=2)

    return trainloader, testloader
