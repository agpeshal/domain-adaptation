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
import resnet
import argparse
import numpy as np

from functions import ReverseLayerF

import mlflow
from mlflow import log_metric, log_params, log_artifacts


class Classify(nn.Module):
    def __init__(self, domains=2, classes=10):
        super(Classify, self).__init__()
        self.fc1 = nn.Linear(64, classes)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, domains)

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
        embedding,
        classifier,
        criterion,
        trainloader,
        testloader,
        device,
        perturb,
        optimizer,
    ):
        self.embedding = embedding
        self.classifier = classifier
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.perturb = perturb
        self.device = device
        self.optimizer = optimizer

    def calc_loss(self, classify, discriminate, labels, domains):

        classification_loss = self.criterion(classify, labels)
        discrimination_loss = self.criterion(discriminate, domains)

        return classification_loss, discrimination_loss

    def train(self, epoch, alpha):
        self.embedding.train()
        self.classifier.train()
        classifier_loss = 0
        discriminator_loss = 0
        net_loss = 0
        correct_d0 = 0
        correct_d1 = 0
        correct_domain = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.trainloader):

            images_d0 = images.to(self.device)
            labels = labels.to(self.device)
            images_d1 = images_d0 + self.perturb * random(images.shape, self.device)

            embeddings_d0 = self.embedding(images_d0)
            embeddings_d1 = self.embedding(images_d1)

            outputs_d0_class, outputs_d0 = self.classifier(embeddings_d0, alpha)
            outputs_d1_class, outputs_d1 = self.classifier(embeddings_d1, alpha)

            # 0 for original, 1 for noisy
            domains = torch.cat(
                (
                    torch.zeros(images.size(0), device=self.device),
                    torch.ones(images.size(0), device=self.device),
                )
            ).long()
            outputs_domains = torch.cat((outputs_d0, outputs_d1))

            loss_class, loss_dis = self.calc_loss(
                outputs_d0_class, outputs_domains, labels, domains
            )
            loss = loss_class + loss_dis
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            classifier_loss += loss_class.item()
            discriminator_loss += loss_dis.item()
            net_loss += loss.item()

            predicted_d0 = outputs_d0_class.argmax(1)
            correct_d0 += predicted_d0.eq(labels).sum().item()

            predicted_d1 = outputs_d1_class.argmax(1)
            correct_d1 += predicted_d1.eq(labels).sum().item()

            predicted_domain = outputs_domains.argmax(1)
            correct_domain += domains.eq(predicted_domain).sum().item()

            total += labels.size(0)

        log_metric("Train/Net_loss", net_loss / (batch_idx + 1), epoch)
        log_metric("Train/Acc_clean", 1.0 * correct_d0 / total, epoch)
        log_metric("Train/Acc_noise", 1.0 * correct_d1 / total, epoch)
        log_metric("Train/Classifier_loss", classifier_loss / (batch_idx + 1), epoch)
        log_metric(
            "Train/Discriminator_loss", discriminator_loss / (batch_idx + 1), epoch
        )
        log_metric("Train/Acc_domain", 1.0 * correct_domain / (2 * total), epoch)

    def test(self, epoch, alpha):
        self.embedding.eval()
        self.classifier.eval()
        classifier_loss = 0
        discriminator_loss = 0
        net_loss = 0
        correct_d0 = 0
        correct_d1 = 0
        correct_domain = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):

                images_d0 = images.to(self.device)
                labels = labels.to(self.device)
                images_d1 = images_d0 + self.perturb * random(images.shape, self.device)

                embeddings_d0 = self.embedding(images_d0)
                embeddings_d1 = self.embedding(images_d1)

                outputs_d0_class, outputs_d0 = self.classifier(embeddings_d0, alpha)
                outputs_d1_class, outputs_d1 = self.classifier(embeddings_d1, alpha)

                # 0 for original, 1 for noisy
                domains = torch.cat(
                    (
                        torch.zeros(images.size(0), device=self.device),
                        torch.ones(images.size(0), device=self.device),
                    )
                ).long()
                outputs_domains = torch.cat((outputs_d0, outputs_d1))

                loss_class, loss_dis = self.calc_loss(
                    outputs_d0_class, outputs_domains, labels, domains
                )
                loss = loss_class + loss_dis
                classifier_loss += loss_class.item()
                discriminator_loss += loss_dis.item()
                net_loss += loss.item()

                predicted_d0 = outputs_d0_class.argmax(1)
                correct_d0 += predicted_d0.eq(labels).sum().item()

                predicted_d1 = outputs_d1_class.argmax(1)
                correct_d1 += predicted_d1.eq(labels).sum().item()

                predicted_domain = outputs_domains.argmax(1)
                correct_domain += domains.eq(predicted_domain).sum().item()

                total += labels.size(0)

        log_metric("Test/Net_loss", net_loss / (batch_idx + 1), epoch)
        log_metric("Test/Acc_clean", 1.0 * correct_d0 / total, epoch)
        log_metric("Test/Acc_noise", 1.0 * correct_d1 / total, epoch)
        log_metric("Test/Classifier_loss", classifier_loss / (batch_idx + 1), epoch)
        log_metric(
            "Test/Discriminator_loss", discriminator_loss / (batch_idx + 1), epoch
        )
        log_metric("Test/Acc_domain", 1.0 * correct_domain / (2 * total), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="resnet56", type=str, help="Model architecture"
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Batch Size")
    parser.add_argument(
        "--epochs", default=150, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--perturb", type=float, default=10.0, help="Magnitude of noise to the input"
    )
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument(
        "--step", default=30, type=int, help="Step size of LR scheduler"
    )
    parser.add_argument(
        "--gamma", default=0.1, type=float, help="Gamma of LR scheduler"
    )

    args = parser.parse_args()

    EXPERIMENT_NAME = "GRL"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding = resnet.__dict__[args.model]()
    # Remove the last layer
    embedding.linear = nn.Identity()
    embedding = embedding.to(device)
    embedding = nn.DataParallel(embedding)
    # checkpoint = torch.load("ckpt.pth", map_location=device)
    # embedding.load_state_dict(checkpoint["net"])
    classifier = Classify().to(device)
    classifier = nn.DataParallel(classifier)
    cuda.benchmark = True

    params = list(embedding.parameters()) + list(classifier.parameters())
    optimizer = Adam(params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    trainloader, testloader = read_vision_dataset("./data", args.batch_size)

    experiment_id = mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(experiment_id=experiment_id):
        log_params(vars(args))
        output_dir = "artifacts"

        criterion = nn.CrossEntropyLoss()
        Obj = Adaptation(
            embedding,
            classifier,
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
        mlflow.pytorch.log_model(embedding, artifact_path="model-base")
        mlflow.pytorch.log_model(classifier, artifact_path="model-head")


if __name__ == "__main__":
    main()
