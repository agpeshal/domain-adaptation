"""Tune CIFAR10 with PyTorch."""
import torch
from torch import optim
from torch.nn.functional import softmax

import argparse

from resnet import ResNet18
from utils import read_vision_dataset, random


def train(net, perturb, batch_size, optimizer, testloader, device, epochs):
    for epoch in range(epochs):
        correct = 0
        total = 0
        confidence = []
        predictions = []
        noise = random(batch_size, device)
        total_loss = 0
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs + perturb * noise[: len(targets)]
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

        print(
            "Epoch {}-- Accuracy: {:.4f}, Loss: {:.4f}".format(
                epoch + 1, correct / total, total_loss / (batch_index + 1)
            )
        )


def main():
    torch.manual_seed(1)
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--batch_size", type=int, default=128, help="Test batch size")
    parser.add_argument(
        "--perturb", type=float, default=0.0, help="Magnitude of noise to the input"
    )
    parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to fine tune"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==> Preparing data..")

    trainloader, testloader = read_vision_dataset("./data", batch_size=args.batch_size)

    print("==> Building model..")

    net = ResNet18(num_classes=10)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    checkpoint = torch.load("ckpt.pth", map_location=device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train(
        net, args.perturb, args.batch_size, optimizer, testloader, device, args.epochs
    )


if __name__ == "__main__":
    main()
