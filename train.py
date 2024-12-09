import argparse
import os
import time
import yaml
import wandb

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from dataloader import listDataset
from val import validate
from model import TreeVision
from utils import load_net_CSRNet, save_checkpoint, prepare_datasets


def pipeline(config):
    """
    The main training pipeline for the TreeVision model.

    Args:
        config (dict): Configuration parameters for training, including paths, hyperparameters, and model settings.

    Returns:
        model (TreeVision): Trained TreeVision model.
    """
    wandb.login()  # Authenticate and connect to Weights & Biases. 

    # Initialize tracking variables
    best_prec1 = 1e6
    is_resume = "allow" if config["pre"] else None

    # Start a new Weights & Biases run. The default wandb project name is TreeVision
    with wandb.init(project="TreeVision", config=config, resume=is_resume, id=config["id"]):
        config = wandb.config  # Access configuration parameters through wandb.config

        # Set up model, data, loss function, and optimizer
        model, train_list, val_list, criterion, optimizer, start_epoch, best_prec1 = make(config)

        # Log model structure and loss function
        wandb.watch(model, criterion, log="all", log_freq=1000)

        # Training loop
        for epoch in range(start_epoch, config.epochs):
            train(train_list, model, criterion, optimizer, epoch + 1, config)  # Train for one epoch

            # Validate the model and log validation loss
            prec1 = validate(val_list, model, criterion, config)
            wandb.log({"val_loss": prec1}, step=epoch + 1)

            # Check if this is the best model so far
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            print(' * Best MAE: {mae:.3f}'.format(mae=best_prec1))

            # Save the model checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.pre,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            }, is_best, task_id=config["id"])

    return model


def make(config):
    """
    Prepares the model, datasets, and optimizer.

    Args:
        config (dict): Configuration dictionary with dataset paths and hyperparameters.

    Returns:
        tuple: Model, training list, validation list, criterion, optimizer, starting epoch, and best validation metric.
    """
    best_prec1 = 1e6
    start_epoch = 0
    seed = time.time()

    # Prepare datasets
    train_list, val_list = prepare_datasets(config.dataset_path, config.dataset_name)

    # Log dataset details
    print("Shape of an image:", plt.imread(train_list[0]).shape)
    print("Training and testing size:", len(train_list), len(val_list))
    
    # Load dataset-specific configurations
    dataset = os.path.basename(config.dataset_path)
    with open(f"{dataset}.yaml", "r") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    torch.cuda.manual_seed(seed)  # Ensure reproducibility

    # Initialize model
    model = TreeVision()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load pre-trained weights for dilated convolutional layers
    load_net_CSRNet(data_dict["Shanghai_B"], model)

    # Define loss function
    criterion = nn.MSELoss(size_average=False).cuda()

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        config.lr,
        betas=(0.9, 0.999),
        weight_decay=config.decay,
    )

    # Load checkpoint if provided
    if config.pre:
        if os.path.isfile(config.pre):
            print(f"=> Loading checkpoint '{config.pre}'")
            checkpoint = torch.load(config.pre)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> Loaded checkpoint '{config.pre}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at '{config.pre}'")

    return model, train_list, val_list, criterion, optimizer, start_epoch, best_prec1


def train(train_list, model, criterion, optimizer, epoch, config):
    """
    Trains the model for one epoch.

    Args:
        train_list (list): Paths to training data.
        model (nn.Module): Model to be trained.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model weights.
        epoch (int): Current epoch number.
        config (dict): Configuration dictionary.
    """
    # Initialize meters for tracking loss and timing
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    print_freq = 100  # Frequency of logging

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        listDataset(
            train_list,
            config.dataset_name,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            train=True,
            batch_size=config.batch_size,
            num_workers=config.workers,
        ),
        batch_size=config.batch_size,
    )

    print(f'Epoch {epoch}, Processed {epoch * len(train_loader.dataset)} samples, LR: {config.lr:.10f}')

    # Switch model to training mode
    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to GPU
        img = img.cuda()
        img = Variable(img)
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        # Compute output and loss
        output = model(img)
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure batch processing time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log training progress
        if i % print_freq == 0:
            print(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})'
            )

    # Log average training loss for the epoch
    wandb.log({"avg_train_loss": losses.avg}, step=epoch)


class AverageMeter:
    """
    Utility class for tracking the average and current value of metrics.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
