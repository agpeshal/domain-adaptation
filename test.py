"""Train CIFAR10 with PyTorch."""
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.functional import softmax

import argparse

from resnet import ResNet18
from tqdm import tqdm
from dataloader import read_vision_dataset
from utils import random
import matplotlib.pyplot as plt


def random(size):

    dist = torch.distributions.normal.Normal(0, 1)
    samples = dist.rsample((size, 3, 32, 32))
    samples = samples / samples.reshape(size, -1).norm(dim=1)[:, None, None, None]
    return samples.to(device)


def test(testloader, net):

    correct = 0
    total = 0
    confidence = []
    predictions = []
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs + args.perturb * noise[: len(targets)]
            # inputs = inputs + args.perturb * random(len(targets))
            outputs = net(inputs)
            prob, predicted = softmax(outputs).max(1)
            confidence.extend(prob.detach().cpu().numpy())
            predictions.extend(predicted.detach().cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        # print(1.*correct/total)


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--model", default="resnet56", type=str, help="Model architecture"
    )
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="Name of the dataset"
    )
    parser.add_argument("--batch", default=128, type=int, help="Batch Size")
    
    args = parser.parse_args()
    print("==> Building model..")
    net = resnet.__dict__[args.model]()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    checkpoint = torch.load("ckpt.pth", map_location=device)
    net.load_state_dict(checkpoint["net"])
    net.eval()

print("Accuracy on Test data {:.4f} ({}/{})".format(correct / total, correct, total))
plt.hist(confidence, density=True, range=(0, 1))
plt.ylim(0, 15)
plt.title("Prediction probability with accuracy :{:.3f}".format(correct / total))
plt.savefig("Scores_d={}.png".format(args.perturb))
plt.show()
# plt.hist(predictions, density=True)
# plt.title("Histogram of predicted labels")
# plt.show()