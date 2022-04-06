from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def getTPTNFPFN(p, g, dim):
    g = 1.0*g
    TP = (p * g).sum(dim=dim)
    FP = (p * (1 - g)).sum(dim=dim)
    FN = ((1 - p) * g).sum(dim=dim)
    TN = ((1 - p) * (1 - g)).sum(dim=dim)
    return torch.stack((TP, TN, FP, FN))


def classLabels2oneHot(classLabels, labels: list):
    '''
    Converts a class label tensor with shape (B, 1, (D,) H, W) to a onehot tensor with shape (B, C, (D,) H, W)
    '''
    shape = list(classLabels.shape)
    shape[1] = max(labels)+1
    onehot = torch.zeros(*tuple(shape))
    return onehot.scatter_(1, classLabels, 1)[:, labels]


class DiceLoss(nn.Module):
    def __init__(self, nDims, batchDice=False, smooth=1e-5):
        super(DiceLoss, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.name = 'Dice Loss'
        self.batchDice = batchDice
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchDice else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        num = (p * g).sum(dim=self.sumDims)
        den = (p + g).sum(dim=self.sumDims)
        dice = ((2. * num + self.smooth) / (den + self.smooth)).mean()
        return 1 - dice


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, nDims, batchDice=False, smooth=1e-5):
        super(GeneralizedDiceLoss, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.name = 'Generalized Dice Loss'
        self.batchDice = batchDice
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchDice else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        w = 1 / (g.sum(dim=self.sumDims) + self.smooth)**2
        num = (w * (p * g).sum(dim=self.sumDims)).sum(dim=-1)
        den = (w * (p + g).sum(dim=self.sumDims)).sum(dim=-1)
        dice = ((2. * num + self.smooth) / (den + self.smooth)).mean()
        return 1 - dice


class FrequencyWeightedDiceLoss(nn.Module):
    def __init__(self, nDims, batchDice=False, smooth=1e-5):
        super(FrequencyWeightedDiceLoss, self).__init__()
        
        self.name = 'Generalized Dice Loss'
        self.batchDice = batchDice
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchDice else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        w = (g.roll(1,1) + g.roll(-1,1)).sum(dim=self.sumDims) / g.sum(self.sumDims).sum(dim=-1, keepdim=True) / 2
        print(w)
        num = (p * g).sum(dim=self.sumDims)
        den = (p + g).sum(dim=self.sumDims)
        dice = (2. * num + self.smooth) / (den + self.smooth)
        return (w*(1 - dice)).sum()


class TverskyLoss(nn.Module):
    def __init__(self, nDims, alpha=0.3, beta=0.7, batchTversky=False, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        
        assert alpha + beta == 1
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.name = 'Tversky Loss'
        self.alpha = alpha
        self.beta = beta
        self.batchTversky = batchTversky
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchTversky else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        tversky = ((TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)).mean()
        return 1- tversky


class GeneralizedTverskyLoss(nn.Module):
    def __init__(self, nDims, alpha=0.3, beta=0.7, batchTversky=False, smooth=1e-5):
        super(GeneralizedTverskyLoss, self).__init__()
        
        assert alpha + beta == 1
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.name = 'Generalized Tversky Loss'
        self.alpha = alpha
        self.beta = beta
        self.batchTversky = batchTversky
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchTversky else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        w = 1 / (g.sum(dim=self.sumDims) + self.smooth)**2
        num = (w * TP).sum(dim=-1)
        den = (w * (TP + self.alpha*FP + self.beta*FN)).sum(dim=-1)
        tversky = ((num + self.smooth) / (den + self.smooth)).mean()
        return 1- tversky


class FocalTverskyLoss(nn.Module):
    def __init__(self, nDims, alpha=0.3, beta=0.7, gamma=0.75, batchTversky=False, smooth=1e-5):
        super(FocalTverskyLoss, self).__init__()

        assert alpha + beta == 1
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.name = 'Focal Tversky Loss'
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batchTversky = batchTversky
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchTversky else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        
        focalTverskyLoss = torch.pow(1 - (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth), self.gamma).mean()
        return focalTverskyLoss


class GeneralizedFocalTverskyLoss(nn.Module):
    def __init__(self, nDims, alpha=0.3, beta=0.7, gamma=0.75, batchTversky=False, smooth=1e-5):
        super(GeneralizedFocalTverskyLoss, self).__init__()
        
        assert alpha + beta == 1
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.name = 'Generalized Focal Tversky Loss'
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.batchTversky = batchTversky
        self.smooth = smooth
        
        self.sumDims = tuple(range(2, nDims+2)) if not batchTversky else (0,) + tuple(range(2, nDims+2))
    
    def forward(self, p, g):
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        w = 1 / (g.sum(dim=self.sumDims) + self.smooth)**2
        num = (w * TP).sum(dim=-1)
        den = (w * (TP + self.alpha*FP + self.beta*FN)).sum(dim=-1)
        generalizedFocalTverskyLoss = torch.pow(1 - (num + self.smooth) / (den + self.smooth), self.gamma).mean()
        return generalizedFocalTverskyLoss


class Dice(nn.Module):
    def __init__(self, nDims, smooth=1.):
        super(Dice, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.abbreviation = 'DSC'
        self.name = 'Dice'
        self.smooth = smooth
        
        self.sumDims = tuple(range(1, nDims+1))
    
    @torch.no_grad()
    def forward(self, p, g):
        p = p.round()
        g = g.round()
        num = (p * g).sum(dim=self.sumDims)
        den = (p + g).sum(dim=self.sumDims)
        dice = ((2. * num + self.smooth) / (den + self.smooth)).mean()
        return dice

    @torch.no_grad()
    def TPTNFPFN(self, TP, TN, FP, FN):
        return 2*TP / (2*TP + FP + FN)


class IoU(nn.Module):
    def __init__(self, nDims, smooth=1.):
        super(IoU, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.abbreviation = 'IoU'
        self.name = 'IoU'
        self.smooth = smooth
        
        self.sumDims = tuple(range(1, nDims+1))
    
    @torch.no_grad()
    def forward(self, p, g):
        p = p.round()
        g = g.round()
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        intersection = TP
        union = TP + FP + FN
        iou = ((intersection + self.smooth) / (union + self.smooth)).mean()
        return iou

    @torch.no_grad()
    def TPTNFPFN(self, TP, TN, FP, FN):
        return TP / (TP + FP + FN)


class Sensetivity(nn.Module):
    def __init__(self, nDims, smooth=1.):
        super(Sensetivity, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.abbreviation = 'Sen'
        self.name = 'Sensetivity'
        self.smooth = smooth
        
        self.sumDims = tuple(range(1, nDims+1))
    
    @torch.no_grad()
    def forward(self, p, g):
        p = p.round()
        g = g.round()
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        sensetivity = ((TP + self.smooth) / (TP + FN + self.smooth)).mean()
        return sensetivity

    @torch.no_grad()
    def TPTNFPFN(self, TP, TN, FP, FN):
        return TP / (TP + FN)


class Specificity(nn.Module):
    def __init__(self, nDims, smooth=1.):
        super(Specificity, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.abbreviation = 'Spe'
        self.name = 'Specificity'
        self.smooth = smooth
        
        self.sumDims = tuple(range(1, nDims+1))
    
    @torch.no_grad()
    def forward(self, p, g):
        p = p.round()
        g = g.round()
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        specificity = ((TN + self.smooth) / (TN + FP + self.smooth)).mean()
        return specificity

    @torch.no_grad()
    def TPTNFPFN(self, TP, TN, FP, FN):
        return TN / (TN + FP)


class F2(nn.Module):
    def __init__(self, nDims, smooth=1.):
        super(F2, self).__init__()
        
        assert nDims in [1, 2, 3]
        self.nDims = nDims
        
        self.abbreviation = 'F2'
        self.name = 'F2'
        self.smooth = smooth
        
        self.sumDims = tuple(range(1, nDims+1))
    
    @torch.no_grad()
    def forward(self, p, g):
        p = p.round()
        g = g.round()
        TP, TN, FP, FN = getTPTNFPFN(p, g, self.sumDims)
        specificity = ((5*TP + self.smooth) / (5*TP + 4*FN + FP + self.smooth)).mean()
        return specificity

    @torch.no_grad()
    def TPTNFPFN(self, TP, TN, FP, FN):
        return 5*TP / (5*TP + FP + 4*FN)


class MetricsCalculator():
    def __init__(self, metrics) -> None:
        self.metrics = metrics
    
    @torch.no_grad()
    def __call__(self, p, g):
        values = torch.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            values[i] = metric(p, g)
        return values
    
    @torch.no_grad()
    def useTP(self, TP, TN, FP, FN):
        values = torch.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            values[i] = metric.TPTNFPFN(TP, TN, FP, FN)
        return values
    
    def getDict(self, lossValue, metricsValues):
        names = ['loss', *[metric.abbreviation for metric in self.metrics]]
        values = ['{:.4f}'.format(lossValue.item()), *['{:.4f}'.format(metricValue.item()) for metricValue in metricsValues]]
        
        d = OrderedDict(zip(names, values))
        return d


if __name__ == '__main__':
    torch.manual_seed(313)
    nDims = 2
    testp = F.softmax(torch.rand((1,3,*(nDims*(32,))), requires_grad=True), dim=1)
    testg = testp.round().detach()
    criterion = DiceLoss(nDims=nDims, batchDice=False)
    print(criterion(testp, testg))
    criterion = GeneralizedDiceLoss(nDims=nDims, batchDice=False)
    print(criterion(testp, testg))
    criterion = FrequencyWeightedDiceLoss(nDims=nDims, batchDice=True)
    print(criterion(testp, testg))
    criterion = TverskyLoss(nDims=nDims, alpha=0.3, beta=0.7, batchTversky=False)
    print(criterion(testp, testg))
    criterion = GeneralizedTverskyLoss(nDims=nDims, alpha=0.3, beta=0.7, batchTversky=False)
    print(criterion(testp, testg))
    criterion = FocalTverskyLoss(nDims=nDims, alpha=0.3, beta=0.7, gamma=0.75, batchTversky=False)
    print(criterion(testp, testg))
    criterion = GeneralizedFocalTverskyLoss(nDims=nDims, alpha=0.3, beta=0.7, gamma=0.75, batchTversky=False)
    print(criterion(testp, testg))
    criterion = nn.CrossEntropyLoss()
    print(criterion(testp, torch.argmax(testg, dim=1)))
    metric = Dice(nDims)
    print(metric(testp[:, 0], testg[:, 0]))
    metric = IoU(nDims)
    print(metric(testp[:, 0], testg[:, 0]))
    metric = Sensetivity(nDims)
    print(metric(testp[:, 0], testg[:, 0]))
    metric = Specificity(nDims)
    print(metric(testp[:, 0], testg[:, 0]))
    metric = F2(nDims)
    print(metric(testp[:, 0], testg[:, 0]))
    # metrics = [
    #     Dice(nDims),
    #     IoU(nDims),
    #     F2(nDims),
    #     Sensetivity(nDims),
    #     Specificity(nDims),
    # ]
    # calc = MetricsCalculator(metrics)
    # print(calc(testp[:, 0], testg[:, 0]))
    # y = torch.LongTensor(1,1,2,2).random_() % 3
    # y_onehot = classLabels2oneHot(y, [0, 1, 2])
    # print(y)
    # print(y_onehot)