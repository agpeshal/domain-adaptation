'''
Performing domain adaption on CIFAR 10
Domain 1: Original input
Domain 2: Random noise added to original inputs
Note that, we train the model from scratch
We use a shallow classifier to discriminate the embeddings
before the fully-connected layer on ResNet18
Goal is train ResNet such that the discriminator fails
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cuda
from torch.optim import Adam, lr_scheduler
from utils import read_vision_dataset, random
from resnet import ResNet18
from tensorboardX import SummaryWriter
import argparse
import numpy as np


class Net(nn.Module):
    def __init__(self, domains=2, classes=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, classes)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, domains)

    def forward(self, x):
        classify = self.fc1(x)
        d = self.fc2(x)
        d = F.relu(d)
        discriminate = self.fc3(d)
        return classify, discriminate


class Adaptation:
    def __init__(self, model, splitter, writer, criterion,
                 trainloader, testloader, device, perturb, optimizer):
        self.model = model
        self.splitter = splitter
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.perturb = perturb
        self.device = device
        self.optimizer = optimizer
        self.writer = writer

    def final_layers(self, features):
        return self.splitter(features)

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
            images_noisy = images +\
                self.perturb * random(images.shape, self.device)

            outputs = self.model(images)
            outputs_noisy = self.model(images_noisy)

            classify, original = self.final_layers(outputs)
            classify_noisy, noisy = self.final_layers(outputs_noisy)

            # 0 for original, 1 for noisy
            domains = torch.cat((torch.zeros(images.size(0), device=self.device),
                                torch.ones(images.size(0), device=self.device))).long()
            discriminate = torch.cat((original, noisy))
            
            loss_class, loss_dis = self.calc_loss(classify, discriminate,
                                                  labels, domains)
            loss = loss_class - alpha * loss_dis
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

        self.writer.add_scalar("Train/Net_loss", net_loss / (batch_idx + 1),
                               epoch)
        self.writer.add_scalar("Train/Acc_clean", 1.*correct / total, epoch)
        self.writer.add_scalar("Train/Acc_noise", 1.*correct_noisy / total,
                               epoch)
        self.writer.add_scalar("Train/Classifier_loss",
                               classifier_loss / (batch_idx + 1), epoch)
        self.writer.add_scalar("Train/Discriminator_loss",
                               discriminator_loss / (batch_idx + 1),
                               epoch)
        self.writer.add_scalar("Train/Acc_domain",
                               1. * correct_domain / (2 * total), epoch)

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
                images_noisy = images +\
                            self.perturb * random(images.shape, self.device)

                outputs = self.model(images)
                outputs_noisy = self.model(images_noisy)

                classify, original = self.final_layers(outputs)
                classify_noisy, noisy = self.final_layers(outputs_noisy)

                domains = torch.cat((torch.zeros(images.size(0), device=self.device),
                                    torch.ones(images.size(0), device=self.device))).long()
                discriminate = torch.cat((original, noisy))
                
                loss_class, loss_dis = self.calc_loss(classify, discriminate,
                                                    labels, domains)
                loss = loss_class - alpha * loss_dis
            
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

        self.writer.add_scalar("Test/Net_loss", net_loss / (batch_idx + 1), epoch)
        self.writer.add_scalar("Test/Acc_clean", 1. * correct / total, epoch)
        self.writer.add_scalar("Test/Acc_noise", 1. * correct_noisy / total,
                               epoch)
        self.writer.add_scalar("Test/Classifier_loss", classifier_loss/(batch_idx + 1), epoch)
        self.writer.add_scalar("Test/Discriminator_loss", discriminator_loss/(batch_idx + 1),
                               epoch)
        self.writer.add_scalar("Test/Acc_domain",
                               1.*correct_domain / (2 * total), epoch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch Size")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Number of training epochs")
    parser.add_argument("--perturb", type=float, default=10.0,
                        help="Magnitude of noise to the input")
    # parser.add_argument("--weight", default=0.01, type=float,
    #                     help="weight given to discriminator")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load("ckpt.pth", map_location=device)

    classifier = ResNet18().to(device)
    classifier = nn.DataParallel(classifier)
    # classifier.load_state_dict(checkpoint["net"])
    discriminator = Net().to(device)
    discriminator = nn.DataParallel(discriminator)
    cuda.benchmark = True

    params = list(classifier.parameters()) + list(discriminator.parameters())
    optimizer = Adam(params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainloader, testloader = read_vision_dataset('./data', args.batch_size)
    writer = SummaryWriter()
    criterion = nn.CrossEntropyLoss()
    Obj = Adaptation(classifier, discriminator, writer, criterion,
                     trainloader, testloader, device, args.perturb,
                     optimizer)

    for epoch in range(1, 1 + args.epochs):
        p = (1e-4 * epoch / args.epochs)
        alpha = 2 / (1 + np.exp(-0.0001 * p)) - 1
        print(f'alpha {alpha:.4f}')
        Obj.train(epoch, alpha)
        Obj.test(epoch, alpha)
        scheduler.step()

    torch.save(classifier.state_dict(), "domain_disc.pth")


if __name__ == "__main__":
    main()
