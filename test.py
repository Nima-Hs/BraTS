import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CustomDataset
from metrics import *
from models import UNet
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    batchSize = 5
    mode = 'WT'
    tag = '1-no-aug'
    testSet = CustomDataset('./test', mode)
    
    testLoader = DataLoader(
        testSet,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)
    
    # writer = SummaryWriter(f'runs/{mode}-test-{tag}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    model = UNet(in_c=4, out_c=2, useBN=True).to(device)
    nDims = 2
    
    metrics = [
        Dice(nDims),
        IoU(nDims),
        F2(nDims),
        Sensetivity(nDims),
        Specificity(nDims)
    ]
    metricsCalculator = MetricsCalculator(metrics)
    
    checkpoints = [os.path.join('./checkpoints', checkpoint) for checkpoint in os.listdir('./checkpoints') if (checkpoint.startswith(mode) & (tag in checkpoint))]
    # print(checkpoints)
    for checkpoint in checkpoints:
        epoch = checkpoint[-2:]
        model.load_state_dict(torch.load(checkpoint)['model'])
        model.eval()
        with torch.no_grad():
            with tqdm(testLoader, unit='batch', desc=f'epoch {epoch} test', dynamic_ncols=True, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as tqdmLoader:
                TPTNFPFN = 0
                metricValues = []
                for i, (input, target) in enumerate(tqdmLoader):
                    input = input.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    
                    prediction = model(input)
                    # # print(prediction.shape,target.shape)
                    # TPTNFPFN += getTPTNFPFN(prediction[:, :1].detach().round(), target[:, :1].detach(), (0,1,2,3))
                    # # print(TPTNFPFN)
                    
                    # if i%(155/batchSize) == 155/batchSize - 1:
                    #     metricValues.append(metricsCalculator.useTP(*(TPTNFPFN)))
                    #     TPTNFPFN = 0
                    #     print(metricValues)
                        # if i != len(tqdmLoader) - 1:
                        #     tqdmLoader.set_postfix(metricsCalculator.getDict())
                        # else:
                        # writer.add_scalars(
                        #     f'Test Dice/{mode}',
                        #     {
                        #         'Test Dice': np.mean(dices)},
                        #     epoch)
                        # tqdmLoader.set_postfix(
                        #     meanDice='{:.4f}'.format(np.mean(dices)),
                        #     stdDice = '{:.4f}'.format(np.std(dices)))
        break