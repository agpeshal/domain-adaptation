"""
Performing domain adaption on CIFAR 10
Domain 1: Original input
Domain 2: Random noise added to original inputs
Note that, we train the model from scratch
We use a shallow classifier to discriminate the embeddings
before the fully-connected layer on ResNet18
Goal is train ResNet such that the discriminator fails
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cuda
from torch.optim import Adam, lr_scheduler
from utils import read_vision_dataset, random
from resnet import ResNet18
import argparse
import numpy as np

from functions import ReverseLayerF

import mlflow
from mlflow import log_metric, log_params, log_artifacts


class Net(nn.Module):
    def __init__(self, domains=2, classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, classes)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, domains)

    def forward(self, x, alpha):
        classify = self.fc1(x)
        rev = ReverseLayerF.apply(x, alpha)
        d = self.fc2(rev)
        d = F.relu(d)
        discriminate = self.fc3(d)
        return classify, discriminate


class Adaptation:
    def __init__(
        self,
        model,
        splitter,
        criterion,
        trainloader,
        testloader,
        device,
        perturb,
        optimizer,
    ):
        self.model = model
        self.splitter = splitter
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.perturb = perturb
        self.device = device
        self.optimizer = optimizer

    def final_layers(self, features, alpha):
        return self.splitter(features, alpha)

    def calc_loss(self, classify, discriminate, labels, domains):

        classification_loss = self.criterion(classify, labels)
        discrimination_loss = self.criterion(discriminate, domains)

        return classification_loss, discrimination_loss

    def train(self, epoch, alpha):
        self.model.train()
        classifier_loss = 0
        discriminator_loss = 0
        net_loss = 0
        correct = 0
        correct_domain = 0
        correct_noisy = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.trainloader):

            images = images.to(self.device)
            labels = labels.to(self.device)
            images_noisy = images + self.perturb * random(images.shape, self.device)

            outputs = self.model(images)
            outputs_noisy = self.model(images_noisy)

            classify, original = self.final_layers(outputs, alpha)
            classify_noisy, noisy = self.final_layers(outputs_noisy, alpha)

            # 0 for original, 1 for noisy
            domains = torch.cat(
                (
                    torch.zeros(images.size(0), device=self.device),
                    torch.ones(images.size(0), device=self.device),
                )
            ).long()
            discriminate = torch.cat((original, noisy))

            loss_class, loss_dis = self.calc_loss(
                classify, discriminate, labels, domains
            )
            loss = loss_class + 0.0 * loss_dis
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            classifier_loss += loss_class.item()
            discriminator_loss += loss_dis.item()
            net_loss += loss.item()

            predicted = classify.argmax(1)
            correct += predicted.eq(labels).sum().item()

            predicted_noisy = classify_noisy.argmax(1)
            correct_noisy += predicted_noisy.eq(labels).sum().item()

            predicted_domain = discriminate.argmax(1)
            correct_domain += domains.eq(predicted_domain).sum().item()

            total += labels.size(0)

        log_metric("Train/Net_loss", net_loss / (batch_idx + 1), epoch)
        log_metric("Train/Acc_clean", 1.0 * correct / total, epoch)
        log_metric("Train/Acc_noise", 1.0 * correct_noisy / total, epoch)
        log_metric("Train/Classifier_loss", classifier_loss / (batch_idx + 1), epoch)
        log_metric(
            "Train/Discriminator_loss", discriminator_loss / (batch_idx + 1), epoch
        )
        log_metric("Train/Acc_domain", 1.0 * correct_domain / (2 * total), epoch)

    def test(self, epoch, alpha):
        self.model.eval()
        classifier_loss = 0
        discriminator_loss = 0
        net_loss = 0
        correct = 0
        correct_domain = 0
        correct_noisy = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):

                images = images.to(self.device)
                labels = labels.to(self.device)
                images_noisy = images + self.perturb * random(images.shape, self.device)

                outputs = self.model(images)
                outputs_noisy = self.model(images_noisy)

                classify, original = self.final_layers(outputs, alpha)
                classify_noisy, noisy = self.final_layers(outputs_noisy, alpha)

                domains = torch.cat(
                    (
                        torch.zeros(images.size(0), device=self.device),
                        torch.ones(images.size(0), device=self.device),
                    )
                ).long()
                discriminate = torch.cat((original, noisy))

                loss_class, loss_dis = self.calc_loss(
                    classify, discriminate, labels, domains
                )
                loss = loss_class + loss_dis

                classifier_loss += loss_class.item()
                discriminator_loss += loss_dis.item()
                net_loss += loss.item()

                predicted = classify.argmax(1)
                correct += predicted.eq(labels).sum().item()

                predicted_noisy = classify_noisy.argmax(1)
                correct_noisy += predicted_noisy.eq(labels).sum().item()

                predicted_domain = discriminate.argmax(1)
                correct_domain += domains.eq(predicted_domain).sum().item()

                total += labels.size(0)

        log_metric("Test/Net_loss", net_loss / (batch_idx + 1), epoch)
        log_metric("Test/Acc_clean", 1.0 * correct / total, epoch)
        log_metric("Test/Acc_noise", 1.0 * correct_noisy / total, epoch)
        log_metric("Test/Classifier_loss", classifier_loss / (batch_idx + 1), epoch)
        log_metric(
            "Test/Discriminator_loss", discriminator_loss / (batch_idx + 1), epoch
        )
        log_metric("Test/Acc_domain", 1.0 * correct_domain / (2 * total), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="Batch Size")
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--perturb", type=float, default=10.0, help="Magnitude of noise to the input"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument(
        "--step", default=20, type=int, help="Step size of LR scheduler"
    )
    parser.add_argument(
        "--gamma", default=0.2, type=float, help="Gamma of LR scheduler"
    )

    args = parser.parse_args()

    EXPERIMENT_NAME = "GRL"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = ResNet18().to(device)
    classifier = nn.DataParallel(classifier)
    # checkpoint = torch.load("ckpt.pth", map_location=device)
    # classifier.load_state_dict(checkpoint["net"])
    discriminator = Net().to(device)
    discriminator = nn.DataParallel(discriminator)
    cuda.benchmark = True

    params = list(classifier.parameters()) + list(discriminator.parameters())
    optimizer = Adam(params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    trainloader, testloader = read_vision_dataset("./data", args.batch_size)

    experiment_id = mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment_id):
        log_params(vars(args))
        output_dir = "artifacts"

        criterion = nn.CrossEntropyLoss()
        Obj = Adaptation(
            classifier,
            discriminator,
            criterion,
            trainloader,
            testloader,
            device,
            args.perturb,
            optimizer,
        )

        for epoch in range(1, 1 + args.epochs):
            p = 1.0 * epoch / args.epochs
            alpha = 2 / (1 + np.exp(-10 * p)) - 1
            print(f"alpha {alpha:.4f}")
            log_metric("alpha", alpha, epoch)
            Obj.train(epoch, alpha)
            Obj.test(epoch, alpha)
            scheduler.step()

        log_artifacts(output_dir, artifact_path="events")
        mlflow.pytorch.log_model(classifier, artifact_path="pytorch-model")


if __name__ == "__main__":
    main()
