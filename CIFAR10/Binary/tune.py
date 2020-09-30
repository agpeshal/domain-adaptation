'''Train CIFAR10 with PyTorch.'''
import torch
from torch import optim
from torch.nn.functional import softmax
import torchvision.transforms as transforms

import argparse

from resnet import ResNet18
from select_classes import get_dataloaders
import matplotlib.pyplot as plt

torch.manual_seed(1)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Test batch size')
parser.add_argument('--perturb', type=float, default=0.0,
                    help='Magnitude of noise to the input')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to fine tune')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Car and Truck dataset
# Test batch size is fixed to 512
trainloader, testloader = get_dataloaders(transform_train, transform_test, batch_size_test=args.batch_size)

print('==> Building model..')

net = ResNet18(num_classes=2)
net = net.to(device)
net = torch.nn.DataParallel(net)
# cudnn.benchmark = True

checkpoint = torch.load('ckpt.pth', map_location=device)
net.load_state_dict(checkpoint['net'])
net.eval()



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
            outputs = net(inputs)
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
        print("Epoch {}-- Accuracy: {:.4f}, Loss: {:.4f}".format(epoch + 1, correct / total, total_loss / (batch_index + 1)))


train(args.epochs)