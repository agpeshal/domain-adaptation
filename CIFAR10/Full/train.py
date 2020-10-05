"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim

import argparse

from resnet import ResNet18
from utils import read_vision_dataset
from tqdm import tqdm


# Training
def train(net, criterion, optimizer, trainloader, device, epoch):
    print("\nEpoch: %d" % epoch)
    net.eval()
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


def test(net, criterion, testloader, best_acc, device, epoch):

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
    if acc > best_acc:
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        torch.save(state, "ckpt.pth")
        best_acc = acc


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--batch", default=128, type=int, help="Batch Size")
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of training epochs"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global best_acc  # best test accuracy
    best_acc = 0
    # Data
    print("==> Preparing data..")
    trainloader, testloader = read_vision_dataset("./data", batch_size=args.batch)
    # Model
    print("==> Building model..")

    net = ResNet18(num_classes=10)
    net = net.to(device)
    net = torch.nn.DataParallel(net)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 1 + args.epochs):
        train(net, criterion, optimizer, trainloader, device, epoch)
        test(net, criterion, testloader, best_acc, device, epoch)
        scheduler.step()


if __name__ == "__main__":
    main()
