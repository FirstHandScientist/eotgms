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
from models.metric import knn, mmd
import ot.gpu
import cudamat
import numpy

import models.dcgan as dcgan
import models.mlp as mlp

#from models.module1 import wsdLoss, entropy, sinkhorn, ccdist
from losses import WassersteinLoss 
########## AUGMENTS FOR BEGIN #########
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')

parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")

parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--sinkgpu', action='store_true', help='Whether to use gpu for sinkhorn calculation')
parser.add_argument('--noM', action='store_true', help='Whether to normalize the cost matrix')
parser.add_argument('--regL', type=float, default=10, help='The wasserstein regularization parameter')

opt = parser.parse_args()
print(opt)

#####################################################################################################################################################PREPARE THE DATA FOR TRAINING#################################################################################################################################################################

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

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
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
#################### configure the constants ####################
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)



################################################## define the distance calculation function 

################################################################################
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.03)
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
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
######################### if use the GPUs for computation, and set the optimizer ####################
#criterion = wsdLoss()
criterion = WassersteinLoss(torch.Tensor([opt.regL]).double(), False)


if opt.cuda:
    netG.cuda()
    #wsdLoss.cuda()

    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

# initialization of cudamat
if opt.sinkgpu:
    cudamat.cublas_init()

normalizeL = torch.Tensor([opt.regL]).double()

SCORE = []

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

        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = input
        #inputvcpu = inputv.cpu().data
        
        ############################
        # (2) Update G network
        ###########################
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        #with torch.no_grad():
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        inputv_fake = fake

        loss = criterion(inputv_fake.view(batch_size, -1), inputv.view(batch_size,-1))        
        loss.backward()
        optimizerG.step()
        #assert False
        gen_iterations += 1
        with torch.no_grad():
            knn_score = knn(inputv.view(batch_size, -1).data, inputv_fake.view(batch_size,-1).data, 1, sqrt=False)
            mmd_score = numpy.array(mmd( inputv.view(batch_size, -1).data, inputv_fake.view(batch_size,-1).data))
    
        tmp_score.append({'iternum':  i,
                          'gLoss': numpy.array(loss.data.cpu()),
                          'knn_score': knn_score.acc.numpy(),
                          'mmd_score': numpy.array(mmd_score) })

        print('[%d/%d][%d/%d][%d] wsdLoss: %f, 1nn_score: %f, mmd_score: %f'%(epoch, opt.niter, i, len(dataloader), gen_iterations, loss.data, knn_score.acc.numpy(), mmd_score))
        if gen_iterations % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu[:64], '{0}/real_samples_{1}.png'.format(opt.experiment, gen_iterations))
            noise.resize_(int(batch_size), nz, 1, 1).normal_(0, 1)    
            noisev = Variable(noise)
            fake = netG(noisev)
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake[:64].data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))

    SCORE.append(tmp_score)
    if epoch %10 ==9:
        # do checkpointing
        torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
  
numpy.save(opt.experiment + '/score_tr', SCORE)   
