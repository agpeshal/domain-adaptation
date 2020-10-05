'''Train CIFAR10 with PyTorch.'''
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.functional import softmax

import argparse

from resnet import ResNet18
from tqdm import tqdm
from utils import read_vision_dataset
import matplotlib.pyplot as plt

torch.manual_seed(1)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Test batch size')
parser.add_argument('--perturb', type=float, default=0.0,
                    help='Magnitude of noise to the input')
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
trainloader, testloader = read_vision_dataset('./data', batch_size=args.batch_size)

# Model
print('==> Building model..')

net = ResNet18(num_classes=10)

net = net.to(device)
net = torch.nn.DataParallel(net)
# cudnn.benchmark = True

checkpoint = torch.load('ckpt.pth', map_location=device)
net.load_state_dict(checkpoint['net'])
net.eval()

correct = 0
total = 0
confidence = []
predictions = []
def random(size):

    dist = torch.distributions.normal.Normal(0, 1)
    samples = dist.rsample((size, 3, 32, 32))
    samples = samples / samples.reshape(size, -1).norm(dim=1)[:, None, None, None]
    return samples.to(device)    

noise = random(args.batch_size)
# print(torch.norm(noise.reshape(args.batch_size, -1), dim=1))
with torch.no_grad():
    for inputs, targets in tqdm(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs + args.perturb * noise[:len(targets)]
        # inputs = inputs + args.perturb * random(len(targets))
        outputs = net(inputs)
        prob, predicted = softmax(outputs).max(1)
        confidence.extend(prob.detach().cpu().numpy())
        predictions.extend(predicted.detach().cpu().numpy())
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # print(1.*correct/total)


print("Accuracy on Test data {:.4f} ({}/{})".format(correct/total, correct, total))
plt.hist(confidence, density=True, range=(0,1))
plt.ylim(0, 15)
plt.title("Prediction probability with accuracy :{:.3f}".format(correct/total))
plt.savefig('Scores_d={}.png'.format(args.perturb))
plt.show()
# plt.hist(predictions, density=True)
# plt.title("Histogram of predicted labels")
# plt.show()