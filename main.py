# A deep learning-based system for survival benefit
# prediction of tyrosine kinase inhibitors and immune
# checkpoint inhibitors in stage IV non-small cell lung
# cancer patients: A multicenter, prognostic study : ESBP
# coding=utf-8

import argparse
import copy
import os
import numpy
import sys

import numpy as np
import torch
import tqdm
from torch import distributed
from torch.utils import data
from torchvision import datasets
from torchvision import transforms
from HDF5_Read import *
from nets import nn
from utils import util
import torch.nn.functional as F
import matplotlib.pyplot as plt
import re
import pandas as pd

import csv

torch.cuda.is_available()


def batch(images, target, model, name, criterion=None):
    images = images.cuda()
    target = target.cuda()
    if criterion:
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            # _, preds = output.topk(1, 1, True, True)
            preds = F.softmax(output, dim=1)
            preds = preds[:, 0]
            # preds = preds.t().squeeze()

        return loss, util.accuracy(output, target, top_k=(1)), preds
    else:
        # return util.accuracy(model(images), target, top_k=(1, 5))
        output = model(images)
        print(output.shape)
        # _, preds = output.topk(1, 1, True, True)
        # preds = preds.t().squeeze()
        preds = F.softmax(output, dim=1)
        preds = preds[:, 0]

        return util.accuracy(output, target, top_k=(1)), preds


def train(args):
    epochs = 250
    batch_size = 20
    util.set_seeds(args.rank)
    model = nn.EfficientNet2(args).cuda()

    model_ori = torch.load("/home/adamdiakite/Documents/ESPS-main/weights/best_pt_OA.pt", map_location='cuda')[
        'model'].float().eval()

    new_model_dict = model_ori.state_dict()

    model.load_state_dict(new_model_dict)

    lr = batch_size * torch.cuda.device_count() * 0.256 / 4096
    optimizer = nn.RMSprop(util.add_weight_decay(model), lr, 0.9, 1e-4, weight_decay=1e-2, momentum=0.9)
    ema = nn.EMA(model)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss().cuda()
    scheduler = nn.StepLR(optimizer)
    amp_scale = torch.cuda.amp.GradScaler()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    Traindataset = H5Dataset(args.input_dir_train)
    train_loader = torch.utils.data.DataLoader(
        Traindataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True)

    for epoch in range(0, epochs):
        if args.distributed:
            sampler.set_epoch(epoch)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 2) % ('epoch', 'loss'))
            bar = tqdm.tqdm(train_loader, total=len(train_loader))
        else:
            bar = train_loader
        model.train()
        top1_train = util.AverageMeter()

        for images, target, name in bar:
            loss, acc_train, preds = batch(images, target, model, name, criterion)
            optimizer.zero_grad()
            amp_scale.scale(loss).backward()
            amp_scale.step(optimizer)
            amp_scale.update()

            ema.update(model)

            torch.cuda.synchronize()
            if args.local_rank == 0:
                bar.set_description(('%10s' + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), loss))

            top1_train.update(acc_train[0].item(), images.size(0))

        scheduler.step(epoch + 1)

    torch.cuda.empty_cache()


def test(input_dir_test, batch_size, model=None):
    model = torch.load('/home/adamdiakite/Documents/ESPS-main/weights/best_pt_OA.pt', map_location='cuda')[
        'model'].float().eval()

    Testdataset = H5Dataset(input_dir_test, False)
    val_loader = torch.utils.data.DataLoader(
        Testdataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    top1 = util.AverageMeter()

    data = []

    with torch.no_grad():
        for images, target, name, tumor in tqdm.tqdm(val_loader, ('%10s') % ('acc@1')):
            acc1, preds = batch(images, target, model, name)
            torch.cuda.synchronize()
            top1.update(acc1[0].item(), images.size(0))

            data.append((str(name), acc1[0].item(), preds, target, tumor))
            print(name, preds, acc1, target)

    # Sort the data based on names
    sorted_data = sorted(data, key=lambda x: natural_sort_key(x[0]))

    # Extract sorted names, accuracies, and tumor areas
    sorted_names = [item[0] for item in sorted_data]
    sorted_accuracy = [item[2] for item in sorted_data]
    sorted_areas = [item[4] for item in sorted_data]

    # Plot the accuracy and tumor area for each name
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # Create a secondary y-axis for tumor areas

    indices = range(len(sorted_names))

    ax1.plot(indices, sorted_accuracy, marker='o', color='tab:blue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2.plot(indices, sorted_areas, marker='o', color='tab:red')
    ax2.set_ylabel('Tumor Area')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Accuracy and Tumor Area for each slice')
    plt.xticks(indices, rotation=90)
    plt.grid(True)
    plt.show()

    # find index of max score slice
    max_accuracy = max(sorted_accuracy)
    # find the volume of the max score slice
    max_surface = max(sorted_areas)

    # find index of average accuracy
    average_accuracy = sum(sorted_accuracy) / len(sorted_accuracy)
    average_surface = sum(sorted_areas) / len(sorted_areas)

    # Print patient ID
    patient_id = input_dir_test[-14:-5]
    print("Patient ID:", patient_id)

    # converting into floats to fit dataframe
    max_accuracy = float(max_accuracy)
    average_accuracy = float(average_accuracy)
    max_surface = float(max_surface)
    average_surface = float(average_surface)
    preds = [float(x) for x in sorted_accuracy]

    dataframe = load_csv_as_dataframe('/home/adamdiakite/Téléchargements/expo_tki.csv')

    # Create a column for each prediction for each slice
    dataframe['max score'] = np.nan
    dataframe['average score'] = np.nan
    dataframe['max surface'] = np.nan
    dataframe['average surface'] = np.nan

    # Update the dataframe with values based on patient ID
    dataframe.loc[dataframe['ID'] == patient_id, 'max score'] = max_accuracy
    dataframe.loc[dataframe['ID'] == patient_id, 'average score'] = average_accuracy
    dataframe.loc[dataframe['ID'] == patient_id, 'max surface'] = max_surface
    dataframe.loc[dataframe['ID'] == patient_id, 'average surface'] = average_surface

    for i, name in enumerate(sorted_names):
        column_name = f"Slice {i}"
        if column_name not in dataframe.columns:
            dataframe[column_name] = np.nan
        dataframe.loc[dataframe['ID'] == patient_id, column_name] = preds[i]

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(dataframe)

    # Save the DataFrame as a CSV file
    output_file = '/home/adamdiakite/Documents/ESPS-main/Test/preds.csv'  # Specify the output file path
    dataframe.to_csv(output_file, index=False)  # Save DataFrame as CSV without index
    print("Dataframe saved at", output_file)

    return acc1


def load_csv_as_array(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


def load_csv_as_dataframe(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe


def natural_sort_key(s):
    # Convert the string to a list of integer and string chunks
    chunks = [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
    # Convert the integers to strings with leading zeros for proper sorting
    return [str(c).zfill(8) if isinstance(c, int) else c for c in chunks]


def natural_sort_key(s):
    # Convert the string to a list of integer and string chunks
    chunks = [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', s)]
    # Convert the integers to strings with leading zeros for proper sorting
    return [str(c).zfill(8) if isinstance(c, int) else c for c in chunks]


def print_parameters(args):
    model = nn.EfficientNet2(args).eval()
    _ = model(torch.zeros(1, 3, 224, 224))
    params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters EfficientNet: {int(params)}')


def benchmark(args):
    shape = (1, 3, 384, 384)
    util.torch2onnx(nn.EfficientNet2(args).export().eval(), shape)
    util.onnx2caffe()
    util.print_benchmark(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--train', action='store_true', default=True)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tf', action='store_true')
    # parser.add_argument('--input_dir_train', type=str, default="/home/adamdiakite/Documents/ESPS-main/Test/Open_Access_Data.hdf5")
    # parser.add_argument('--input_dir_val', type=str, default="/home/adamdiakite/Documents/ESPS-main/Test/Open_Access_Data.hdf5t")
    parser.add_argument('--input_dir_test', type=str,
                        default="/home/adamdiakite/Documents/ESPS-main/Test/patient_data/2-21-0001.hdf5")

    args = parser.parse_args()
    args.distributed = False
    args.rank = 0
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.rank = torch.distributed.get_rank()

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    if args.local_rank == 0:
        print_parameters(args)

    if args.benchmark:
        benchmark(args)

    test("/home/adamdiakite/Documents/ESPS-main/Test/patient_data/2-21-0001.hdf5", batch_size=1)

    if args.test:
        test(args)


if __name__ == '__main__':
    main()