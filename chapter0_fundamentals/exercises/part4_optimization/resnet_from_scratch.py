#%%
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


device = torch.device("mps" if torch.has_mps else "cpu")


#%%
def get_cifar10(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    cifar_testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(1, len(cifar_trainset), subset))
        # cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))

    return cifar_trainset, cifar_testset

#%%
cifar_trainset, cifar_testset = get_cifar10(subset=1)
cifar_trainloader = DataLoader(cifar_trainset, batch_size=128, shuffle=True)
cifar_testloader = DataLoader(cifar_testset, batch_size=128, shuffle=False)

#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#%%
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(num_classes)


    def _make_layer(self, block, out_channels, blocks, stride = 1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels

        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.softmax

        return x

#%%
num_classes = 10
num_epochs = 10
batch_size = 128
learning_rate = 0.01
loss_fn = nn.CrossEntropyLoss()
total_step = len(cifar_trainloader)

#%%
model = ResNet(ResidualBlock, [2, 2, 2, 2], num_classes).to(device)

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

#%%
@torch.no_grad()
def validate():
    correct = 0
    total = 0
    for images, labels in cifar_testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Valid Accuracy on the {total} validation images: {100 * correct / total}")


#%%
def train():
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(cifar_trainloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}")

        # Validation
        validate()


#%%
train()

#%%
validate()