from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils import data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from pytorch_dataloader import ECGDataset
from resnet_time_frequency import resnet18
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import click
import argparse


def run(lead1, lead2):
    leads = [int(lead1), int(lead2)]
    writer = SummaryWriter()
    data_dir = "/om2/user/sadhana/time-frequency-data/"
    ecg_dataset = {
        "train": ECGDataset(csv_file="total_train.csv", root_dir=data_dir, leads=leads),
        "val": ECGDataset(csv_file="total_val.csv", root_dir=data_dir, leads=leads),
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            ecg_dataset[x], batch_size=4, shuffle=True, num_workers=4
        )
        for x in ["train", "val"]
    }
    dataset_sizes = {x: len(ecg_dataset[x]) for x in ["train", "val"]}
    class_names = ecg_dataset["train"].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs - 1))
            print("-" * 10)
            print(leads)
            # Each epoch has a training and validation phase
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.type(torch.LongTensor)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(
                    "{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )
                writer.add_scalar("Loss/" + phase, epoch_loss, epoch)

                # deep copy the model
                if phase == "val" and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()
            y_pred = []
            y_true = []

            # iterate over test data
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                output = model(inputs)  # Feed Network

                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth

            cf_matrix = confusion_matrix(y_true, y_pred)
            print(cf_matrix)
            print("===========================")
        time_elapsed = time.time() - since
        print(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print("Best val Acc: {:4f}".format(best_acc))
        # load best model weights
        model.load_state_dict(best_model_wts)

        y_pred = []
        y_true = []

        # iterate over test data
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            output = model(inputs)  # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)  # Save Prediction

            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

        cf_matrix = confusion_matrix(y_true, y_pred)
        FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)
        FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
        TP = np.diag(cf_matrix)
        TN = cf_matrix.values.sum() - (FP + FN + TP)

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP / (TP + FN)
        # Specificity or true negative rate
        TNR = TN / (TN + FP)
        # Precision or positive predictive value
        PPV = TP / (TP + FP)
        # Negative predictive value
        NPV = TN / (TN + FN)
        # Fall out or false positive rate
        FPR = FP / (FP + TN)
        # False negative rate
        FNR = FN / (TP + FN)
        # False discovery rate
        FDR = FP / (TP + FP)
        print("TPR", TPR, "FPR", FPR)

        # Overall accuracy
        ACC = (TP + TN) / (TP + FP + FN + TN)
        print(cf_matrix)
        return model

    model = resnet18(num_channels=len(leads))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    summed = 11125 + 4846 + 1868 + 779
    weights = 1 / torch.tensor([11125 + 4846, 1868 + 779])
    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion.cuda()
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=17)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l1")
    parser.add_argument("-l2")
    args = parser.parse_args()
    run(int(args.l1), int(args.l2))


if __name__ == "__main__":
    main()
