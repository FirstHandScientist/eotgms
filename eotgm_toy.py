import os, sys

sys.path.append(os.getcwd())

import random
import numpy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import scipy.stats as stats 


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.stats import moment
from models.mlp import MLP_G, MLP_D
from losses import WassersteinLoss, WassersteinLossL1, emdLoss
import cudamat
from models.toy_data import inf_train_gen
#from utils import Logger
from models.metric import knn, mmd, fid
matplotlib.rcParams.update({'font.size': 18})
torch.manual_seed(1)

class setting():
    def __init__(self):
        self.MODE = 'eotgm'  # wgan or wgan-gp or eotgm
        self.DATASET = '4gaussians'  # 8gaussians, 25gaussians, swissroll
        self.REL_DIR = 'toy_result/toymodel' +'/' +self.MODE+ self.DATASET + '/'

        self.VARIANCE = 1.0
        #RANGE = 10 
        self.RANGE = 7
        self.BIAS = 0
        self.NZ = 2


        self.REGULATION = True
        # normalizeL = torch.Tensor([0.5]) this is used for 8/9 gaussian
        self.normalizeL = torch.Tensor([0.5])
        self.LATENT = 'gaussian' # 'gaussian' or 'uniform'
        self.DIM = 256  # Model dimensionality
        self.FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
        self.BATCH_SIZE = 256  # Batch size
        self.ITERS = 2000  # how many generator iterations to
opt = setting()
SCORE = []
use_cuda = True

cudamat.cublas_init()
if not os.path.exists(opt.REL_DIR):
    print("Making dir: {}".format(opt.REL_DIR))
    os.system('mkdir -p {0}'.format(opt.REL_DIR))




# ==================Definition Start======================

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.04)
        m.bias.data.fill_(0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.03)
        m.bias.data.fill_(0.1)



def density_estimation(m1, m2):
    X, Y = np.mgrid[-opt.RANGE:opt.RANGE:100j, -opt.RANGE:opt.RANGE:100j]                                                     
    positions = np.vstack([X.ravel(), Y.ravel()])                                                       
    values = np.vstack([m1, m2])                                                                        
    kernel = stats.gaussian_kde(values)                                                                 
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

frame_index = [0]
def generate_image(true_dist, fake_dist):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    true_dist_v = autograd.Variable(torch.Tensor(true_dist).cuda() if use_cuda else torch.Tensor(true_dist))
    samples = fake_dist.cpu().data.numpy()
    #samples = netG(noisev, true_dist_v).cpu().data.numpy()

    plt.clf()

    #x = y = np.linspace(-opt.RANGE, opt.RANGE, N_POINTS)
    #plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='red', marker='+', alpha=1)
    X, Y, Z = density_estimation(true_dist[:, 0], true_dist[:, 1])
    plt.contour(X, Y, Z, colors = 'red', alpha=0.5, linewidth=0.8)
    if not opt.FIXED_GENERATOR:
        plt.scatter(samples[:, 0], samples[:, 1], marker='o', facecolors="None",edgecolors='b', alpha =1)
        Xf, Yf, Zf = density_estimation(samples[:, 0], samples[:, 1])
        plt.contour(Xf, Yf, Zf, colors = 'b', alpha=0.5, linewidth=0.8)
    if opt.LATENT=="uniform":
        plt.savefig(opt.REL_DIR + 'un_' + 'frame' + str(frame_index[0]) + '.jpg', dpi =1000)
    else:
        plt.savefig( opt.REL_DIR + 'frame' + str(frame_index[0]) + '.jpg', dp1=1000)

    frame_index[0] += 1

# ==================Definition End======================
netG = MLP_G(2, opt.NZ, 1, opt.DIM, 1)


#netG = Generator()
#netD = Discriminator()


netG.apply(weights_init)


print( netG)


if use_cuda:
    netG = netG.cuda()


optimizerG = optim.Adam(netG.parameters(), lr=5e-4, betas=(0.5, 0.9))

if opt.REGULATION:
    wassLoss = WassersteinLoss(opt.normalizeL, False)
    emdLoss = emdLoss()
else:
    wassLoss = emdLoss(False)
one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda()
    mone = mone.cuda()

data = inf_train_gen(opt.DATASET, opt.BATCH_SIZE, opt.BIAS, opt.VARIANCE)

TURNrl = True
for iteration in range(opt.ITERS):
 
    ############################
    # Update G network
    ###########################
    
    
    netG.zero_grad()
    wassLoss.zero_grad()
    
    _data = next(data)
    real_data = torch.Tensor(_data)
    if use_cuda:
        real_data = real_data.cuda()
    real_data_v = autograd.Variable(real_data)

    if opt.LATENT == 'gaussian':
        noise = torch.randn(opt.BATCH_SIZE, opt.NZ)
    else:
        noise = torch.rand(opt.BATCH_SIZE, opt.NZ) - 0.5

    if use_cuda:
        noise = noise.cuda()
    noisev = autograd.Variable(noise)
    fake = netG(noisev, real_data_v)
    # with torch.no_grad():
    #     moms = numpy.absolute(moment(fake.cpu().numpy(),axis=None, moment=range(1,5))-moment(real_data_v.cpu().numpy(),axis=None, moment=range(1,5)))
    #     logger.writer.add_scalar('err_m1',moms[0], iteration)
    #     logger.writer.add_scalar('err_m2',moms[1], iteration)
    #     logger.writer.add_scalar('err_m3',moms[2], iteration)
    #     logger.writer.add_scalar('err_m4',moms[3], iteration)
    if iteration < 20:
        gLoss = emdLoss(fake,real_data_v)
    else:      
        gLoss = wassLoss(fake, real_data_v)
        
    gLoss.backward()
    optimizerG.step()
    #logger.log(0, gLoss.data, iteration)
    with torch.no_grad():
        knn_score = knn(real_data.data, fake.data, 1, sqrt=False)
        mmd_score = numpy.array(mmd(real_data.data, fake.data))
        fid_score = numpy.array(fid(real_data.cpu().data, fake.cpu().data))
    #print('[%d / %d], G_cost: %f'%(iteration,ITERS, gLoss.data))
        #logger.writer.add_scalar('score',knn_score.acc, iteration)

    print('[%d / %d], G_cost: %f, 1nn_score: %f, mmd: %f, fid: %f'%(iteration, opt.ITERS, gLoss.data, knn_score.acc, mmd_score, fid_score))
    
    SCORE.append({'iter': iteration, 
                  'gloss': gLoss.data.numpy(), 
                  'knnscore': knn_score.acc.numpy(), 
                  'mmdscore':mmd_score,
                  'fidscore': fid_score
                 })
    # if gLoss<20 and TURNrl:
    #     with torch.no_grad():
    #         wassLoss.update_reg(2)
    #         TURNrl = False

    if iteration % 100 == 99:
        generate_image(_data, fake)

numpy.save('{0}score_tr{1}'.format(opt.REL_DIR, opt.DATASET), SCORE)    
#logger.close()
with open(opt.REL_DIR+ 'configuration.text', 'w') as f:
    f.write('%s'%(opt.__dict__))
