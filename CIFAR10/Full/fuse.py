'''Train CIFAR10 with PyTorch.'''
import torch
from torch import optim
from torch.nn.functional import softmax

import argparse

from resnet import ResNet18
from tqdm import tqdm
from utils import read_vision_dataset
import matplotlib.pyplot as plt

torch.manual_seed(1)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Test batch size')
parser.add_argument('--perturb', type=float, default=0.0,
                    help='Magnitude of noise to the input')
parser.add_argument('--weight', default=0.5, type=float,
                    help='Weight of the pretrained model')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to fine tune')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')

trainloader, testloader = read_vision_dataset('./data', batch_size=args.batch_size)

print('==> Building model..')

net = ResNet18(num_classes=10)
net = net.to(device)
net = torch.nn.DataParallel(net)

# load pre-trained model
pretrained = ResNet18(num_classes=10)
pretrained = pretrained.to(device)
pretrained = torch.nn.DataParallel(pretrained)
checkpoint = torch.load('ckpt.pth', map_location=device)
pretrained.load_state_dict(checkpoint['net'])
pretrained.eval()



def random(size):

    dist = torch.distributions.normal.Normal(0, 1)
    samples = dist.rsample((size, 3, 32, 32))
    samples = samples / samples.reshape(size, -1).norm(dim=1)[:, None, None, None]
    return samples.to(device)


optimizer = optim.SGD(net.parameters(), lr=1e-5,
                      momentum=0.9, weight_decay=5e-4)
def train(epochs):
    for epoch in range(epochs):
        correct = 0
        total = 0
        confidence = []
        predictions = []
        noise = random(args.batch_size)
        total_loss = 0
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs + args.perturb * noise[:len(targets)]
            # inputs = inputs + args.perturb * random(len(targets))
            outputs = args.weight * pretrained(inputs)\
                      + (1 - args.weight) * net(inputs)
            prob, predicted = softmax(outputs).max(1)
            optimizer.zero_grad()
            loss = torch.dot(prob, 1 - prob) / (targets.size(0))
            loss.backward()
            optimizer.step()
            confidence.extend(prob.detach().cpu().numpy())
            predictions.extend(predicted.detach().cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()

        # print("Epoch {}: Loss: {:.4f}".format(epoch + 1, total_loss/(batch_index + 1)))
        # print("Epoch {}: Accuracy on Test data {:.4f} ({}/{})".format(epoch + 1, correct / total, correct, total))
        print("Epoch {}-- Accuracy: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, correct / total, total_loss / (batch_index + 1)))


train(args.epochs)