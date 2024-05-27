import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import CIFAR10
import tqdm
import wandb

from modeling.diffusion import DiffusionModel
from modeling.training import train_step, train_epoch, generate_samples
from modeling.unet import UnetModel


@pytest.fixture
def train_dataset():
    transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(
        "./data",
        train=True,
        download=True,
        transform=transforms,
    )
    return dataset


@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(device, train_dataset):
    # note: you should not need to increase the threshold or change the hyperparameters
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32),
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )
    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=5e-4)
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    x, _ = next(iter(dataloader))
    loss = None
    for i in range(50):
        loss = train_step(ddpm, x, optim, device)
    assert loss < 0.5


@pytest.mark.parametrize(["device"],[["cuda"]])
def test_training(device, train_dataset, epochs = 50):
    config = {"lr":5e-4, "batch_size":4, "epochs":epochs, "dataset":"CIFAR-100"}
    
    ddpm = DiffusionModel(
        eps_model=UnetModel(3, 3, hidden_size=32), 
        betas=(1e-4, 0.02),
        num_timesteps=1000,
    )

    ddpm.to(device)

    optim = torch.optim.Adam(ddpm.parameters(), lr=config['lr'])

    dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    wandb.init(
        project="efficient_learning_hw2",
        config = config
    )
    
    for i in range(epochs):
        loss = train_epoch(ddpm, dataloader, optimizer=optim, device = device)
        wandb.log({"loss":loss})

    generate_samples(ddpm, device, '')

    wandb.finish()
