"""
Tune CIFAR10 using only the test data without labels
Add random perturbations of fixed norm to test images
and tune the weights of pretrained model to reduce the
entropy of the probability corresponding to the predicted
label.
"""
import torch
from torch import optim
from torch.nn.functional import softmax

import argparse
import resnet
from dataloader import read_vision_dataset
from utils import random

import mlflow
from mlflow import log_metric, log_params, log_artifacts


def train(net, perturb, optimizer, testloader, device, epoch):
    correct = 0
    total = 0
    # confidence = []
    # predictions = []
    total_loss = 0
    for batch_index, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        images = images + perturb * random(images.shape, device)
        outputs = net(images)
        prob, predicted = softmax(outputs).max(1)
        optimizer.zero_grad()
        loss = torch.dot(prob, 1 - prob) / (labels.size(0))
        loss.backward()
        optimizer.step()
        # confidence.extend(prob.detach().cpu().numpy())
        # predictions.extend(predicted.detach().cpu().numpy())
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

    print(
        "Epoch {}-- Accuracy: {:.4f}, Loss: {:.4f}".format(
            epoch, correct / total, total_loss / (batch_index + 1)
        )
    )
    log_metric("Accuracy_target", 1.0 * correct / total, epoch)
    log_metric("Loss_target", total_loss / (batch_index + 1), epoch)


def test(net, testloader, device, epoch):
    correct = 0
    total = 0
    # confidence = []
    # predictions = []
    total_loss = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            prob, predicted = softmax(outputs).max(1)
            loss = torch.dot(prob, 1 - prob) / (labels.size(0))
            # confidence.extend(prob.detach().cpu().numpy())
            # predictions.extend(predicted.detach().cpu().numpy())
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()

        print(
            "Epoch {}-- Accuracy: {:.4f}, Loss: {:.4f}".format(
                epoch, correct / total, total_loss / (batch_index + 1)
            )
        )
        log_metric("Accuracy_source", 1.0 * correct / total, epoch)
        log_metric("Loss_source", total_loss / (batch_index + 1), epoch)


def main():

    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
    parser.add_argument(
        "--model", default="resnet56", type=str, help="Model architecture"
    )
    parser.add_argument(
        "--dataset", type=str, default="CIFAR10", help="Name of the dataset"
    )
    parser.add_argument("--batch", type=int, default=128, help="Test batch size")
    parser.add_argument(
        "--perturb", type=float, default=10.0, help="Magnitude of noise to the input"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to fine tune"
    )
    args = parser.parse_args()

    EXPERIMENT_NAME = "Entropy Minimization"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("==> Preparing data..")
    trainloader, testloader = read_vision_dataset(
        "./data", batch_size=args.batch, dataset=args.dataset
    )

    print("==> Building model..")
    net = resnet.__dict__[args.model]()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    checkpoint = torch.load("ckpt.pth", map_location=device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    experiment_id = mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment_id):
        log_params(vars(args))
        for epoch in range(1, args.epochs + 1):

            train(
                net,
                args.perturb,
                optimizer,
                testloader,
                device,
                epoch,
            )
            test(net, testloader, device, epoch)

        mlflow.pytorch.log_model(net, artifact_path="tuned model")


if __name__ == "__main__":
    main()
