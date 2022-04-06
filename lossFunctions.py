import torch
import torch.nn as nn
import torch.nn.functional as F


def batchDiceLoss(target: torch.Tensor, prediction: torch.Tensor, smooth: float=0.0001) -> torch.Tensor:
    intersection = (target * prediction).sum(dim=(0, 2, 3))
    denom = target.sum(dim=(0, 2, 3)) + prediction.sum(dim=(0, 2, 3))
    dice = (2*intersection + smooth)/(denom + smooth)
    W = ((target.sum(dim=(0, 2, 3)) + smooth)/(target.sum() + smooth)).flip(0)
    return ((1-dice)*W).sum()

def diceLoss(target: torch.Tensor, prediction: torch.Tensor, smooth: float=0.0001) -> torch.Tensor:
    intersection = (target * prediction).sum(dim=(2, 3))
    denom = target.sum(dim=(2, 3)) + prediction.sum(dim=(2, 3))
    dice = (2*intersection + smooth)/(denom + smooth)
    W = (target.sum(dim=(2, 3))/target.sum(dim=(1, 2, 3)).view(-1, 1)).fliplr()
    return ((1-dice)*W).sum(dim=1).mean()

def batchDiceRound(target: torch.Tensor, prediction: torch.Tensor, smooth: float=0.0001) -> torch.Tensor:
    intersection = (target * prediction.detach().round()).sum(dim=(0, 2, 3))
    denom = target.sum(dim=(0, 2, 3)) + prediction.detach().round().sum(dim=(0, 2, 3))
    dice = (2*intersection + smooth)/(denom + smooth)
    return dice

def diceRound(target: torch.Tensor, prediction: torch.Tensor, smooth: float=0.0001) -> torch.Tensor:
    intersection = (target * prediction.detach().round()).sum(dim=(2, 3))
    denom = target.sum(dim=(2, 3)) + prediction.detach().round().sum(dim=(2, 3))
    dice = (2*intersection + smooth)/(denom + smooth)
    return dice.mean(dim=0)

def erosion(input: torch.Tensor, kernel_size: tuple) -> torch.Tensor:
    if len(input.size()) - 2 != len(kernel_size):
        raise ValueError(f'input spatial dimensions = ({len(input.size())-2}), kernel dimensions = ({len(kernel_size)})')
    padding = ()
    for s in kernel_size:
        padding += tuple(2*[(s-1)//2])
    input_unfold = torch.cat((F.pad(input[:, 0:1], pad=padding, value=0), F.pad(input[:, 1:], pad=padding, value=1)), dim=1)
    input_unfold = F.unfold(input_unfold, kernel_size=kernel_size)
    input_unfold = input_unfold.view(input.size(0), input.size(1), -1, input_unfold.size(2))
    return torch.min(input_unfold, dim=2).values.view(input.size())

def edge(input: torch.Tensor, kernel_size: tuple=(3, 3)) -> torch.Tensor:
    return 1.0*input - erosion(1.0*input, kernel_size=kernel_size)

def getNomDenom(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    prediction = prediction.detach().round()
    Nom = 2*(target * prediction).sum(dim=(0,2,3))
    Denom = target.sum(dim=(0,2,3)) + prediction.sum(dim=(0,2,3))
    return Nom, Denom

if __name__ == '__main__':
    a = torch.tensor([[[
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 1., 1., 0.],
        [0., 0., 1., 1., 1., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.]]]], requires_grad=True)
    a = torch.cat((a, 1-a), dim=1)
    b = torch.rand((1,1,8,8))
    print(b)
    # print(edge(a, (3,3)))
    loss = lambda target, prediction: (diceLoss(target, prediction))
    print(loss(a,b))