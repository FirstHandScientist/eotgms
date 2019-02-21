import torch
import torch.nn as nn
import torch.nn.functional as F
import cudamat
from torch.nn import PairwiseDistance
import ot
import ot.gpu
import numpy

class WassersteinLoss(nn.Module):
    """
    Define the wasserstein distance of two batch of empirical data samples as loss
    """
    def __init__(self, entropy_reg, noCost=True):
        super(WassersteinLoss, self).__init__()
        self.entropy_reg = entropy_reg
        self.normalize_cost = noCost
 
    
    def ccdist(self, input1, input2):
        'Calculate the pair-wise distance between two empirical samples'
        output = torch.empty(len(input1),len(input2))
        pdist = PairwiseDistance(p=2)

        for i in range( len(input1) ):
            dup_input1 = input1[i].repeat(len(input2),1)
            output[i] = pdist(dup_input1, input2)
            
        return output
    def update_reg(self, assignment=torch.Tensor([1])):
        self.normalize_cost = assignment

    def forward(self, sample1, sample2):
        costMatrix = self.ccdist(sample1, sample2).double()**2
        if self.normalize_cost:
            with torch.no_grad():
                maxCost = costMatrix.data.max()
            costMatrix = costMatrix/maxCost

        with torch.no_grad():
            cost_gpu = cudamat.CUDAMatrix(costMatrix.data.numpy())
            tranport, opt_log = ot.gpu.sinkhorn(numpy.ones((sample1.size(0),))/sample1.size(0), numpy.ones((sample2.size(0),))/sample2.size(0), cost_gpu, float(self.entropy_reg.numpy()[0]), log=True )
            transport = torch.DoubleTensor(tranport)
        
        distance = torch.sum(transport*costMatrix)
        return distance

class WassersteinLossL1(WassersteinLoss):
    """
    Define the wasserstein distance of two batch of empirical data samples as loss, Norm 1 case
    """
    def __init__(self, entropy_reg, noCost=True):
        #super(WassersteinLossL1, self).__init__(entropy_reg, noCost=True)
        super(WassersteinLossL1, self).__init__(entropy_reg, noCost=True)

    def forward(self, sample1, sample2):
        costMatrix = self.ccdist(sample1, sample2).double()
        
        if self.normalize_cost:
            with torch.no_grad():
                maxCost = costMatrix.data.max()
            costMatrix = costMatrix/maxCost

        with torch.no_grad():
            cost_gpu = cudamat.CUDAMatrix(costMatrix.data.numpy())
            tranport, opt_log = ot.gpu.sinkhorn(numpy.ones((sample1.size(0),))/sample1.size(0), numpy.ones((sample2.size(0),))/sample2.size(0), cost_gpu, float(self.entropy_reg.numpy()[0]), log=True )
            transport = torch.DoubleTensor(tranport)
        
        distance = torch.sum(transport*costMatrix)
        return distance

class emdLoss(nn.Module):
    """
    Define the earth mover distance of two batch of empirical data samples as loss
    """
    def __init__(self, noCost=True):
        super(emdLoss, self).__init__()
        self.normalize_cost = noCost
    
    def ccdist(self, input1, input2):
        'Calculate the pair-wise distance between two empirical samples'
        output = torch.empty(len(input1),len(input2))
        pdist = PairwiseDistance(p=2)

        for i in range( len(input1) ):
            dup_input1 = input1[i].repeat(len(input2),1)
            output[i] = pdist(dup_input1, input2)
            
        return output

    def forward(self, sample1, sample2):
        costMatrix = self.ccdist(sample1, sample2).double()**2
        if self.normalize_cost:
            costMatrix = costMatrix/costMatrix.max()
        with torch.no_grad():
            tranport, opt_log = ot.emd(numpy.ones((sample1.size(0),))/sample1.size(0), numpy.ones((sample2.size(0),))/sample2.size(0), costMatrix.cpu().data.numpy(), log=True )
            transport = torch.DoubleTensor(tranport)
        
        distance = torch.sum(transport*costMatrix)
        return distance


# class ContrastiveLoss(nn.Module):
#     """
#     Contrastive loss
#     Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
#     """

#     def __init__(self, margin):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, target, size_average=True):
#         distances = (output2 - output1).pow(2).sum(1)  # squared distances
#         losses = 0.5 * (target.float() * distances +
#                         (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
#         return losses.mean() if size_average else losses.sum()


# class TripletLoss(nn.Module):
#     """
#     Triplet loss
#     Takes embeddings of an anchor sample, a positive sample and a negative sample
#     """

#     def __init__(self, margin):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative, size_average=True):
#         # distance_positive = ((anchor - positive).pow(2)/(anchor + positive)).sum(1)  # .pow(.5)
#         # distance_negative = ((anchor - negative).pow(2)/(anchor + positive)).sum(1)  # .pow(.5)
        
#         #losses = (distance_positive - distance_negative)/anchor.size()[1] + self.margin
#         distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#         distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        
#         losses = (distance_positive - distance_negative) + self.margin
        
#         return losses.sum()

class TripletWassersteinLoss(nn.Module):
    """
    Triplet loss where distance is wasserstein distance instead of euclidean distanc 
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, entropy_reg, normalizeM=True, cutoff = None, ratio = 1):
        super(TripletWassersteinLoss, self).__init__()
        self.margin = margin
        self.reg = entropy_reg
        self.normalize_cost = normalizeM
        self.cutoff = cutoff
        self.ratio = ratio
 

    def forward(self, anchor, positive, negative, new_ratio):
        self.ratio = new_ratio
        wsd = WassersteinLoss(self.reg, self.normalize_cost)
        
        distance_positive = wsd(anchor, positive)
        distance_negative = wsd(anchor, negative)
        if self.cutoff is "leaky":
            losses = F.leaky_relu((distance_positive - distance_negative) + self.margin)
        elif self.cutoff is "relu":
            losses = F.relu((distance_positive - distance_negative) + self.margin)
        else:
            losses = distance_positive - self.ratio*distance_negative

        return losses.sum(), distance_positive, distance_negative



