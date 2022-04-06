import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import augmentation as aug
import utils
from datasets import CustomDataset
from lossFunctions import batchDiceLoss, batchDiceRound, diceRound, edge, diceLoss
from models import UNet
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    trainingPaths, validationPaths, testPaths = utils.separateData(
        './MICCAI_BraTS2020_TrainingData', 0.1, 0.1, ('flair', 't1', 't1ce', 't2', 'seg'))

    transforms = aug.Compose([
        aug.RandomVerticalFlip(0.5),
        aug.RandomVerticalTranslation(0.1),
        aug.RandomHorizontalTranslation(0.1),
        aug.RandomVerticalScale((0.9, 1.1)),
        aug.RandomHorizontalScale((0.9, 1.1)),
        aug.RandomVerticalShear(0.05),
        aug.RadnomHorizontalShear(0.05),
        aug.RandomRotation2D(15),
        aug.ElasticTransform(alpha=600, sigma=30),
    ], repeat=10)

    utils.createDataset(trainingPaths, './training', precision=np.single, batchSize=4, deleteEmptySlices=True, augment=True, transforms=transforms)
    # utils.createDataset(trainingPaths, './training', precision=np.single, batchSize=4, deleteEmptySlices=True, augment=False)
    utils.createDataset(validationPaths, './validation', precision=np.single, batchSize=4, deleteEmptySlices=True, augment=False)
    utils.createDataset(testPaths, './test', precision=np.single, batchSize=4, deleteEmptySlices=False, augment=False)

    mode = 'WT'
    tag = '1-no-aug'
    batchSize = 10

    trainingSet = CustomDataset('./training', mode)
    validationSet = CustomDataset('./validation', mode)

    trainingLoader = DataLoader(
        trainingSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    validationLoader = DataLoader(
        validationSet,
        batch_size=batchSize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

    writer = SummaryWriter(f'runs/{mode}-{tag}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_c=4, out_c=2, useBN=True).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        threshold=1e-2,
        verbose=True)
    # lossFunction = lambda target, prediction: (batchDiceLoss(target, prediction) + batchDiceLoss(edge(target), edge(prediction)))/2
    lossFunction = lambda target, prediction: batchDiceLoss(edge(target), edge(prediction))
    # lossFunction = batchDiceLoss

    model.load_state_dict(torch.load(
        './checkpoints/WT-1-no-aug-edge-loss-epoch-53')['model'])
    optimizer.load_state_dict(torch.load(
        './checkpoints/WT-1-no-aug-edge-loss-epoch-53')['optimizer'])
    scheduler.load_state_dict(torch.load(
        './checkpoints/WT-1-no-aug-edge-loss-epoch-53')['scheduler'])
    # for g in optimizer.param_groups:
    #     g['lr'] = 1e-4


    for epoch in range(54, 90):
        model.train()
        with tqdm(trainingLoader, unit='batch', desc=f'epoch {epoch} training', dynamic_ncols=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdmLoader:
            avgBatchLoss = 0
            avgRoundSliceDice = 0
            avgRoundBatchDice = 0
            for i, (input, target) in enumerate(tqdmLoader):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                prediction = model(input)

                loss = lossFunction(target, prediction)
                avgBatchLoss += loss/len(tqdmLoader)
                roundSliceDice = diceRound(target[:, :1], prediction[:, :1])
                avgRoundSliceDice += roundSliceDice/len(tqdmLoader)
                roundBatchDice = batchDiceRound(
                    target[:, :1], prediction[:, :1])
                avgRoundBatchDice += roundBatchDice/len(tqdmLoader)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i != len(tqdmLoader) - 1:
                    tqdmLoader.set_postfix(
                        Loss='{:.4f}'.format(loss.item()),
                        roundSliceDice='{:.4f}'.format(roundSliceDice.item()),
                        roundBatchDice='{:.4f}'.format(roundBatchDice.item()))
                else:
                    tqdmLoader.set_postfix(
                        avgBatchLoss='{:.4f}'.format(avgBatchLoss.item()),
                        avgRoundSliceDice='{:.4f}'.format(
                            avgRoundSliceDice.item()),
                        avgRoundBatchDice='{:.4f}'.format(avgRoundBatchDice.item()))
                # break
        scheduler.step(avgBatchLoss)

        model.eval()
        with torch.no_grad():
            with tqdm(validationLoader, unit='batch', desc=f'epoch {epoch} validation', dynamic_ncols=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdmLoader:
                avgValBatchLoss = 0
                avgValRoundSliceDice = 0
                avgValRoundBatchDice = 0
                for i, (input, target) in enumerate(tqdmLoader):
                    input = input.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    prediction = model(input)

                    valLoss = lossFunction(target, prediction)
                    avgValBatchLoss += valLoss/len(tqdmLoader)
                    valRoundSliceDice = diceRound(
                        target[:, :1], prediction[:, :1])
                    avgValRoundSliceDice += valRoundSliceDice/len(tqdmLoader)
                    valRoundBatchDice = batchDiceRound(
                        target[:, :1], prediction[:, :1])
                    avgValRoundBatchDice += valRoundBatchDice/len(tqdmLoader)

                    if i != len(tqdmLoader)-1:
                        tqdmLoader.set_postfix(
                            valLoss='{:.4f}'.format(valLoss.item()),
                            valRoundSliceDice='{:.4f}'.format(
                                valRoundSliceDice.item()),
                            valRoundBatchDice='{:.4f}'.format(valRoundBatchDice.item()))
                    else:
                        tqdmLoader.set_postfix(
                            avgValBatchLoss='{:.4f}'.format(
                                avgValBatchLoss.item()),
                            avgValRoundSliceDice='{:.4f}'.format(
                                avgValRoundSliceDice.item()),
                            avgValRoundBatchDice='{:.4f}'.format(avgValRoundBatchDice.item()))

        torch.save(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()},
            f'./checkpoints/{mode}-{tag}-epoch-' + f'{epoch}'.zfill(2))
        writer.add_scalars(
            f'Batch Loss/{mode}',
            {
                'Train Batch Loss': avgBatchLoss,
                'Validation Batch Loss': avgValBatchLoss},
            epoch)
        writer.add_scalars(
            f'Round Slice Dice/{mode}',
            {
                'Train Round Slice Dice': avgRoundSliceDice,
                'Validation Round Slice Dice': avgValRoundSliceDice},
            epoch)
        writer.add_scalars(
            f'Round Batch Dice/{mode}',
            {
                'Train Round Batch Dice': avgRoundBatchDice,
                'Validation Round Batch Dice': avgValRoundBatchDice},
            epoch)
