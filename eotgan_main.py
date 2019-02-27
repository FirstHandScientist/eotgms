from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os

import ot.gpu
import cudamat
import numpy

import models.dcgan as dcgan
import models.mlp as mlp
from models.metric import knn, mmd


#from models.module1 import wsdLoss, entropy, sinkhorn, ccdist
from models.networks import EmbeddingNet, EmbeddingNetTD, TripletNet
from losses import WassersteinLoss, TripletWassersteinLoss

########## AUGMENTS FOR BEGIN #########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='mnist | cifar10 ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--sfn', type=int, default=2, help='size of features used for similarity mapping output')
parser.add_argument('--emb_net', required = True, help='Chose the net used for embedding_net')
parser.add_argument('--equ_ratio', type = float, default = 1.0, help='Chose the equilibrium for to control')

parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')

parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.00005')
parser.add_argument('--lrS', type=float, default=0.0001, help='learning rate for similarity function, default=0.001')
parser.add_argument('--margin', type=float, default=5, help='the margin value for triplet loss')

parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netF', default='', help="path to f (to continue training)")

parser.add_argument('--netS', required= True, help="siamese | triplet | rawWass")

parser.add_argument('--Siters', type=int, default=1, help='number of netS iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--sinkgpu', action='store_true', help='Whether to use gpu for sinkhorn calculation')
parser.add_argument('--noM', action='store_true', help='Whether to normalize the cost matrix')
parser.add_argument('--regL', type=float, default=10, help='The wasserstein regularization parameter')
parser.add_argument('--rawWass', action='store_true', help = 'Whether to calculate wasserstein distance by raw pixel feature')

opt = parser.parse_args()
print(opt)

#####################################################################################################################################################PREPARE THE DATA FOR TRAINING#################################################################################################################################################################

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir -p {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

####################################################################################################
## Configure the networks ###
####################################################################################################
####################################################################################################
#################### configure the constants ####################
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)
sfn = int(opt.sfn)
Siters = int(opt.Siters)
equ_ratio = float(opt.equ_ratio)
######################################## define the distance calculation function 

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

##################################INPUT- LATEN Z -- ##################################################################
input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)


noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

# configure siamese network if activated
margin = opt.margin

if opt.netS =='triplet':
    if opt.emb_net == 'EmbeddingNet':
        embedding_net = EmbeddingNet(nc)
    elif opt.emb_net == 'EmbeddingNetTD':
        embedding_net = EmbeddingNetTD(opt.imageSize, nc, ndf, ngpu, sfn)
    
    current_ratio = torch.DoubleTensor([1.])
    netS = TripletNet(embedding_net)
    #loss_fn = TripletLoss(margin)
    loss_fn = TripletWassersteinLoss(margin,
                                     torch.Tensor([opt.regL]).double(),
                                     normalizeM=False,
                                     cutoff = 'relu', ratio=current_ratio)
print(netS)
netS.apply(weights_init)    

if opt.netF != '': # load checkpoint if needed
    netS.load_state_dict(torch.load(opt.netF))
print(netS)

############################################################################################################################################################################################################################################################################################################

######################### if use the GPUs for computation, and set the optimizer ####################
#criterion = wsdLoss()


if opt.cuda:
    netG.cuda()
    #wsdLoss.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    if opt.netS is not 'rawWass':
        netS.cuda()

# setup optimizer
if opt.adam:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    if opt.netS is not 'rawWass':
        optimizerS = optim.Adam(netS.parameters(), lr=opt.lrS, betas=(opt.beta1, 0.999))
else:
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)
    if opt.netS is not 'rawWass':
        optimizerS = optim.RMSprop(netS.parameters(), lr=opt.lrS)

# initialization of cudamat
if opt.sinkgpu:
    cudamat.cublas_init()

normalizeL = torch.Tensor([opt.regL]).double()

# data_iter = iter(dataloader)
# data = data_iter.next()

wassLoss = WassersteinLoss(normalizeL, False)



SCORE = []
equilibrium_rate = 1e-3

gen_iterations = 0
for epoch in range(opt.niter):
    data_iter = iter(dataloader)
    i = 0
    tmp_score = []
    while i < len(dataloader):
        ############################
        # (1) sample the empirical data first
        ###########################
        data = data_iter.next()
        i += 1
        # fetch the real data
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        half_batch_size = int(batch_size/2)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = input

        
        ############################
        # (2) Update S network
        ###########################
        for p in netS.parameters():
            p.requires_grad = True # to avoid computation
                
        j = 0
        while j < Siters and i < len(dataloader):
            j += 1
            ## sample the fake inputs
            with torch.no_grad():
                noise.resize_(int(batch_size/2), nz, 1, 1).normal_(0, 1)    
                noisev = Variable(noise)
                fake = netG(noisev)
                inputv_fake = Variable(fake.data)

            netS.zero_grad()
            loss_fn.zero_grad()
            out_archor, out_positive, out_negative  = netS(inputv[0:int(batch_size/2)],inputv[int(batch_size/2):batch_size], inputv_fake)
            errS, tmp_lp, tmp_ln = loss_fn(out_archor, out_positive, out_negative, new_ratio = 1)
        
            errS.backward()
            optimizerS.step()

        # print('[%d/%d][%d/%d][%d] similarityLoss: %f'
        #        % (epoch, opt.niter, i, len(dataloader), j, errS.data))
        
        ##########################################
        # update G network
        #########################################
        for p in netS.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        netS.zero_grad()
        wassLoss.zero_grad()

        noise.resize_(int(batch_size/2), nz, 1, 1).normal_(0, 1)    
        noisev = Variable(noise)
        fake = netG(noisev)
        fea1, fea2, fea3 = netS(inputv[0:half_batch_size],inputv[half_batch_size:batch_size], fake)
        
        # errS = loss_fn(fea1, fea2, fea3)
        
        # errS.backward()
        with torch.no_grad():
            sLoss = wassLoss(fea1, fea2)
        gLoss = wassLoss(fea3, fea1)
            
        gLoss.backward()
        optimizerG.step()
        #assert False
        gen_iterations += 1
        
        with torch.no_grad():
            knn_score = knn(fea1.view(half_batch_size, -1).data, fea3.view(half_batch_size,-1).data, 1, sqrt=False)
            mmd_score = numpy.array(mmd(fea1.view(half_batch_size, -1).data, fea3.view(half_batch_size,-1).data))
    
        tmp_score.append({'iternum':  i,
                      'sLoss': numpy.array(gLoss.data.cpu()),
                      'gLoss': numpy.array(gLoss.data.cpu()),
                      'knn_score': knn_score.acc.numpy(),
                      'mmd_score': numpy.array(mmd_score) })
        
        #print('[%d/%d][%d/%d][%d] wsdLossPD: %f, wsdLossPP: %f, simLoss: %f, score: (%f, %f), kt: %f'
        #     % (epoch, opt.niter, i, len(dataloader), gen_iterations, gLoss.data, sLoss.data, errS.data, knn_score.acc.numpy(), mmd_score, current_ratio))
        print('[%d/%d][%d/%d][%d] wsdLossPD: %f, wsdLossPP: %f, simLoss: %f, score: (%f, %f)'
             % (epoch, opt.niter, i, len(dataloader), gen_iterations, gLoss.data, 
                sLoss.data, errS.data, knn_score.acc.numpy(), mmd_score))
        if gen_iterations % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu[:64], '{0}/real_samples.png'.format(opt.experiment))
            
            noise.resize_(int(batch_size/2), nz, 1, 1).normal_(0, 1)    
            noisev = Variable(noise)
            fake = netG(noisev)
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake[:64].data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
    #SCORE.append({'iter': iteration, 'gloss': gLoss.data.numpy(), 'knnscore': knn_score.acc.numpy(), 'mmdscore':mmd_score})
    SCORE.append(tmp_score)
    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    if epoch %10 ==9:
        # do checkpointing

        torch.save(netS.state_dict(), '{0}/netS_epoch_{1}.pth'.format(opt.experiment, epoch))

numpy.save(opt.experiment + '/score_tr', SCORE)   
