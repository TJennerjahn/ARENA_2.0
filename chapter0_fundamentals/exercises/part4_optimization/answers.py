#%%
import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
import pandas as pd
import torch as t
from torch import Tensor, optim
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from typing import Callable, Iterable, Tuple, Optional, Type
from jaxtyping import Float
from dataclasses import dataclass
from tqdm.notebook import tqdm
from pathlib import Path
import numpy as np
from IPython.display import display, HTML

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
def pathological_curve_loss(x: t.Tensor, y: t.Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


if MAIN:
    plot_fn(pathological_curve_loss)


#%%
def opt_fn_with_sgd(fn: Callable, xy: t.Tensor, lr=0.02, momentum=0.99, n_iters: int = 100):
    '''
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters, 2). The (x,y) BEFORE each step. So out[0] is the starting point.
    '''
    assert xy.requires_grad, "xy must be a tensor with requires_grad=True"
    out = t.zeros((n_iters, 2))
    xy = xy.requires_grad_(True)
    optimizer = t.optim.SGD([xy], lr=lr, momentum=momentum)
    for i in range(n_iters):
        out[i] = xy.detach()
        loss = fn(xy[0], xy[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Final Pair: ", out[-1])


    return out

# opt_fn_with_sgd(pathological_curve_loss, t.tensor([0.0, 0.0]), lr=0.001, momentum=0.98, n_iters=100)

#%%
if MAIN:
    points = []

    optimizer_list = [
        (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
        (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params['lr'], momentum=params['momentum'])

        points.append((xys, optimizer_class, params))

    plot_fn_with_points(pathological_curve_loss, points=points)


#%%
class SGD:
    def __init__(
        self, 
        params: Iterable[t.nn.parameter.Parameter], 
        lr: float, 
        momentum: float = 0.0, 
        weight_decay: float = 0.0
    ):
        '''Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay
        self.gs = [t.zeros_like(param) for param in params]
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:

        for i, (g, param) in enumerate(zip(self.gs, self.params)):
            new_g = param.grad
            if self.lmda != 0:
                new_g += (self.lmda * param)
            if self.mu != 0 and self.t > 1:
                new_g += (self.mu * g)
            self.params[i] -= self.lr * new_g
            self.gs[i] = new_g

        self.t += 1


    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"



if MAIN:
    tests.test_sgd(SGD)


#%%
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        '''Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html

        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.lmda = weight_decay
        self.mu = momentum

        self.vs = [t.zeros_like(param) for param in params]
        self.bs = [t.zeros_like(param) for param in params]

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (p, b, v) in enumerate(zip(self.params, self.bs, self.vs)):
            new_g = p.grad
            if self.lmda != 0:
                new_g += (self.lmda * p)
            new_v = self.alpha * v + (1 - self.alpha) * new_g ** 2
            if self.mu > 0:
                new_b = self.mu * b + (new_g / (t.sqrt(new_v) + self.eps))
                self.params[i] -= self.lr * new_b
                self.bs[i] = new_b
            else:
                self.params[i] -= self.lr * (new_g / (t.sqrt(new_v) + self.eps))
            self.vs[i] = new_v
        

    def __repr__(self) -> str:
        return f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"



if MAIN:
    tests.test_rmsprop(RMSprop)


#%%
class Adam:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.ms = [t.zeros_like(param) for param in params]
        self.vs = [t.zeros_like(param) for param in params]
        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (m, v, param) in enumerate(zip(self.ms, self.vs, self.params)):
            new_g = param.grad
            if self.lmda != 0:
                new_g += (self.lmda * param)
            new_m = self.beta1 * m + (1 - self.beta1) * new_g
            new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
            m_hat =  new_m / (1 - self.beta1 ** self.t)
            v_hat = new_v / (1 - self.beta2 ** self.t)

            self.params[i] -= self.lr * (m_hat / (t.sqrt(v_hat) + self.eps))
            self.ms[i] = new_m
            self.vs[i] = new_v
        
        self.t += 1

            

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adam(Adam)

#%%
class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        '''Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        '''
        params = list(params) # turn params into a list (because it might be a generator)
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay

        self.ms = [t.zeros_like(param) for param in params]
        self.vs = [t.zeros_like(param) for param in params]

        self.t = 1

    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, (m, v) in enumerate(zip(self.ms, self.vs)):
            new_g = self.params[i].grad
            self.params[i] -= self.lr * self.lmda * self.params[i]
            new_m = self.beta1 * m + (1 - self.beta1) * new_g
            new_v = self.beta2 * v + (1 - self.beta2) * new_g.pow(2)
            m_hat =  new_m / (1 - self.beta1 ** self.t)
            v_hat = new_v / (1 - self.beta2 ** self.t)

            self.params[i] -= self.lr * (m_hat / (t.sqrt(v_hat) + self.eps))
            self.ms[i] = new_m
            self.vs[i] = new_v

        self.t += 1


    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



if MAIN:
    tests.test_adamw(AdamW)


#%%
def opt_fn(fn: Callable, xy: t.Tensor, optimizer_class, optimizer_hyperparams: dict, n_iters: int = 100):
    '''Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    '''

    assert xy.requires_grad, "xy must be a tensor with requires_grad=True"
    out = t.zeros((n_iters, 2))
    xy = xy.requires_grad_(True)
    optimizer = optimizer_class([xy], **optimizer_hyperparams)
    for i in range(n_iters):
        out[i] = xy.detach()
        loss = fn(xy[0], xy[1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Final Pair ({optimizer_class.__name__}): ", out[-1])

    return out




#%%
def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * (x - x_mean) ** 2) / (2 * x_sig**2)
    y_exp = (-1 * (y - y_mean) ** 2) / (2 * y_sig**2)
    return norm * t.exp(x_exp + y_exp)

def neg_trimodal_func(x, y):
    z = -bivariate_gaussian(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)
    return z


if MAIN:
    plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2))

#%%
def rosenbrocks_banana_func(x: t.Tensor, y: t.Tensor, a=1, b=100) -> t.Tensor:
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


if MAIN:
    plot_fn(rosenbrocks_banana_func, x_range=(-2, 2), y_range=(-1, 3), log_scale=True)


#%%
#%%
if MAIN:
    landscape = pathological_curve_loss
    # landscape = neg_trimodal_func
    # landscape = rosenbrocks_banana_func
    # landscape = bivariate_gaussian
    points = []

    optimizer_list = [
        (SGD, {"lr": 0.03, "momentum": 0.99}),
        (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
        (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
        (AdamW, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    ]

    for optimizer_class, params in optimizer_list:
        xy = t.tensor([2.5, 2.5], requires_grad=True)
        xys = opt_fn(landscape, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params, n_iters=100)
        points.append((xys, optimizer_class, params))

    plot_fn_with_points(landscape, points=points)

#%%
def get_cifar(subset: int = 1):
    cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=IMAGENET_TRANSFORM)
    if subset > 1:
        cifar_trainset = Subset(cifar_trainset, indices=range(0, len(cifar_trainset), subset))
        cifar_testset = Subset(cifar_testset, indices=range(0, len(cifar_testset), subset))
    return cifar_trainset, cifar_testset


if MAIN:
    cifar_trainset, cifar_testset = get_cifar()

    imshow(
        cifar_trainset.data[:15],
        facet_col=0,
        facet_col_wrap=5,
        facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
        title="CIFAR-10 images",
        height=600
    )


#%%
@dataclass
class ResNetTrainingArgs():
    batch_size: int = 64
    epochs: int = 3
    optimizer: Type[t.optim.Optimizer] = t.optim.Adam
    learning_rate: float = 1e-3
    n_classes: int = 10
    subset: int = 10

#%%
class ResNetTrainer:
    def __init__(self, args: ResNetTrainingArgs):
        self.args = args
        self.model = get_resnet_for_feature_extraction(args.n_classes).to(device)
        self.optimizer = args.optimizer(self.model.out_layers[-1].parameters(), lr=args.learning_rate)
        self.trainset, self.testset = get_cifar(subset=args.subset)
        self.logged_variables = {"loss": [], "accuracy": []}

    def _shared_train_val_step(self, imgs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = self.model(imgs)
        return logits, labels

    def training_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
        logits, labels = self._shared_train_val_step(imgs, labels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss

    @t.inference_mode()
    def validation_step(self, imgs: Tensor, labels: Tensor) -> t.Tensor:
        logits, labels = self._shared_train_val_step(imgs, labels)
        classifications = logits.argmax(dim=1)
        n_correct = t.sum(classifications == labels)
        return n_correct

    def train_dataloader(self):
        self.model.train()
        return DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        self.model.eval()
        return DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=True)

    def train(self):
        progress_bar = tqdm(total=self.args.epochs * len(self.trainset) // self.args.batch_size)
        accuracy = t.nan

        for epoch in range(self.args.epochs):

            # Training loop (includes updating progress bar)
            for imgs, labels in self.train_dataloader():
                loss = self.training_step(imgs, labels)
                self.logged_variables["loss"].append(loss.item())
                desc = f"Epoch {epoch+1}/{self.args.epochs}, Loss = {loss:.2f}, Accuracy = {accuracy:.2f}"
                progress_bar.set_description(desc)
                progress_bar.update()

            # Compute accuracy by summing n_correct over all batches, and dividing by number of items
            accuracy = sum(self.validation_step(imgs, labels) for imgs, labels in self.val_dataloader()) / len(self.testset)

            self.logged_variables["accuracy"].append(accuracy.item())


#%%
args = ResNetTrainingArgs()
trainer = ResNetTrainer(args)
trainer.train()
plot_train_loss_and_test_accuracy_from_trainer(trainer, title="Training ResNet on MNIST data")
