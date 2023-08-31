#%%
import math
import os

import torchinfo; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import Tensor, optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional, Type
from jaxtyping import Float
from dataclasses import dataclass
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML
import wandb

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_optimization"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))
os.chdir(section_dir)

from plotly_utils import bar, imshow, plot_train_loss_and_test_accuracy_from_trainer
from part3_resnets.solutions import IMAGENET_TRANSFORM, ResNet34, get_resnet_for_feature_extraction
from part4_optimization.utils import plot_fn, plot_fn_with_points
import part4_optimization.tests as tests

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

#%%
# MNIST Conv Net
class MNISTModel(nn.Module):
    def __init__(self, model_size_factor: int = 1):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(1, math.floor(32 * (math.sqrt(2) ** model_size_factor)), 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(math.floor(32 * (math.sqrt(2) ** model_size_factor)), math.floor(64 * (math.sqrt(2) ** model_size_factor)), 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(math.floor(64 * (math.sqrt(2) ** model_size_factor)) * 7 * 7, math.floor(128 * (math.sqrt(2) ** model_size_factor))),
            nn.Linear(math.floor(128 * (math.sqrt(2) ** model_size_factor)), 10),
        )
    def forward(self, x):
        return self.sequential(x)

#%%
model = MNISTModel().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 5
batch_size = 64

#%%
print(model)


#%%
summary = torchinfo.summary(model, (1, 1, 28, 28))
print(summary)

#%%
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_mnist(subset: int = 1):
    '''Returns MNIST training data, sampled by the frequency given in `subset`.'''
    mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=MNIST_TRANSFORM)

    if subset > 1:
        mnist_trainset = Subset(mnist_trainset, indices=range(0, len(mnist_trainset), subset))
        # mnist_testset = Subset(mnist_testset, indices=range(0, len(mnist_testset), subset))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

#%%
@dataclass
class MNISTTrainingArgs():
    '''
    Defining this class implicitly creates an __init__ method, which sets arguments as 
    given below, e.g. self.batch_size = 64. Any of these arguments can also be overridden
    when you create an instance, e.g. args = ConvNetTrainingArgs(batch_size=128).
    '''
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    model_size_factor: int = 1
    subset: int = 10

@dataclass
class MNISTTrainingArgsWandb(MNISTTrainingArgs):
    wandb_project: str = "mnist_scaling"
    wandb_name: str = None

#%%
class MNISTTrainer:
    def __init__(self, args: MNISTTrainingArgsWandb):
        self.args = args
        self.args.learning_rate = 1/math.sqrt(64 * (math.sqrt(2) ** self.args.model_size_factor) * 7 * 7)
        self.model = MNISTModel(self.args.model_size_factor).to(device)
        self.optimizer = args.optimizer(self.model.parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_mnist(subset=args.subset)
        self.t = 0
        print("learning_rate", args.learning_rate)
        wandb.init(project=args.wandb_project, name=args.wandb_name, config={**args.__dict__, "model_size": sum(p.numel() for p in self.model.parameters() if p.requires_grad), "data_size": len(self.trainset)})
        wandb.watch(self.model, log="all", log_freq=20)
        # wandb.log({"batch_size": args.batch_size, "epochs": args.epochs, "learning_rate": args.learning_rate, "data_size": len(self.trainset), "model_size": sum(p.numel() for p in self.model.parameters() if p.requires_grad)})

    def _shared_train_val_step(self, imgs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        return logits, labels

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        logits, labels = self._shared_train_val_step(imgs, labels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.t += 1
        return loss

    @t.inference_mode()
    def validation_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        logits, labels = self._shared_train_val_step(imgs, labels)
        classifications = logits.argmax(dim=1)
        n_correct = t.sum(classifications == labels)
        return n_correct

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

    def train(self):
        progress_bar = tqdm(total=self.args.epochs * len(self.trainset) // self.args.batch_size)
        accuracy = 0

        wandb.log({"accuracy": accuracy}, step=self.t)
        for epoch in range(self.args.epochs):

            # Training loop (includes updating progress bar)
            for imgs, labels in self.train_dataloader():
                loss = self.training_step(imgs, labels)
                wandb.log({"loss": loss.item()}, step=self.t)
                desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}"
                progress_bar.set_description(desc)
                progress_bar.update()

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.testset)
            wandb.log({"accuracy": accuracy.item()}, step=self.t)
        
        wandb.finish()

    
#%%

if MAIN:
    subsets = [1,2,4,8,16]
    model_size_factors = [1,2,3,4,5]
    for subset in subsets:
        for model_size_factor in model_size_factors:
            print(f"Run with subset={subset}, model_size_factor={model_size_factor}")
            args = MNISTTrainingArgsWandb(batch_size=128, epochs=1, subset=subset, model_size_factor=model_size_factor)
            trainer = MNISTTrainer(args)
            trainer.train()
    # plot_train_loss_and_test_accuracy_from_trainer(trainer, title="MNIST Training Loss and Test Accuracy")
