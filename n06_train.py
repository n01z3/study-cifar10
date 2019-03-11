import os
from tqdm import tqdm
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil

from n04_dataset import get_loaders
from n05_model import Net

import logging


def get_logger(log_filepath):
    logger = logging.getLogger(log_filepath)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_checkpoint(state, is_best, filename='weights/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'weights/model_best.pth.tar')


def main():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, 'logs.txt')

    logger = get_logger(log_filepath)

    os.makedirs('weights', exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()

    net = Net().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    trainloader, validloader, _ = get_loaders(batch_size=128)

    best_acc = 0.0
    for epoch in range(120):
        for i, batch in enumerate(tqdm(trainloader, total=len(trainloader)), 0):
            # get the inputs
            inputs = batch['image'].to(device)
            labels = batch['y'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        correct = 0
        total = 0
        with torch.no_grad():
            for batch in validloader:
                inputs = batch['image'].to(device)
                labels = batch['y'].to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        logger.info(f'epoch:{epoch} | acc:{acc}')

        is_best = False
        if acc > best_acc:
            best_acc = acc
            is_best = True
        save_checkpoint(net.state_dict(), is_best=is_best)


if __name__ == '__main__':
    main()
