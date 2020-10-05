"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim

import argparse

import resnet
from dataloader import read_vision_dataset
from tqdm import tqdm
from torchsummary import summary

# Training
def train(net, criterion, optimizer, trainloader, device, epoch):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(
        "Loss: %.3f | Acc: %.3f%% (%d/%d)"
        % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
    )


def test(net, criterion, testloader, device, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(
            "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)
        )

    # Save checkpoint.
    acc = 100.0 * correct / total

    return acc


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--model", default="resnet56", type=str, help="Model architecture"
    )
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="Name of the dataset"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch", default=128, type=int, help="Batch Size")
    parser.add_argument(
        "--epochs", default=200, type=int, help="Number of training epochs"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0  # best test accuracy
    # Data
    print("==> Preparing data..")
    trainloader, testloader = read_vision_dataset(
        "./data", batch_size=args.batch, dataset=args.dataset
    )
    # Model
    print("==> Building model..")

    net = resnet.__dict__[args.model]()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    summary(net, (3, 32, 32))

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 1 + args.epochs):
        train(net, criterion, optimizer, trainloader, device, epoch)
        acc = test(net, criterion, testloader, device, epoch)
        scheduler.step()

        if acc > best_acc:
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": acc,
                "epoch": epoch,
            }
            torch.save(state, "ckpt.pth")
            best_acc = acc


if __name__ == "__main__":
    main()
