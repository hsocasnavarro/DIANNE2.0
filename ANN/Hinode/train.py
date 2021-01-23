#!/scratch/hsocas/miniconda3/bin/python3.7
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import zarr
from torch.autograd import Variable
import time
from tqdm import tqdm
import model
import argparse
import platform
import nvidia_smi
import sys
import os
import pathlib
import scipy.io as io

class Dataset(torch.utils.data.Dataset):
    Weights=[1.,1e-2,1e-2,1e-1]
    [nx,ny,nlam]=[0,0,0]
    def __init__(self):
        super(Dataset, self).__init__()

        print("Reading database with models...")
        tmp = io.readsav('params.idl')

        params = np.transpose(tmp['params'], axes=(2,1,0))
        [self.nx,self.ny,npars]=params.shape

        self.phys = params.reshape((self.nx*self.ny, npars))
        print('Number of parameters={}'.format(npars))

        # Disambiguation
        ibx=6
        ind = np.where(self.phys[:,ibx] < 0.0)[0]
        if len(ind) > 0: print("Disambiguating {} profiles".format(len(ind)))
        self.phys[ind,ibx] = -self.phys[ind,ibx]
        self.phys[ind,ibx+1] = -self.phys[ind,ibx+1]
        
        normalization = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e5]) 
        normalization_mean = np.array([5e3, 5e3, 5e3, 5e3, 5e3, 0, 0, 0, 0])
        
        self.phys = (self.phys - normalization_mean[None,:]) / normalization[None,:]

        self.n_training, self.n_par = self.phys.shape

        print("Reading database with Stokes profiles...")
        tmp=io.readsav('database.prof.idl')
        [self.nlam,self.ny2,self.nx2]=tmp['stki'].shape
        self.stokes=np.zeros((self.nx2,self.ny2,self.nlam,4))
        self.stokes[:,:,:,0]=np.transpose(tmp['stki'], axes=(2,1,0) )
        self.stokes[:,:,:,1]=np.transpose(tmp['stkq'], axes=(2,1,0) )
        self.stokes[:,:,:,2]=np.transpose(tmp['stku'], axes=(2,1,0) )
        self.stokes[:,:,:,3]=np.transpose(tmp['stkv'], axes=(2,1,0) )     

        # print("Normalizing Stokes parameters...")

        #[self.nx2,self.ny2,self.nlam,nstokes]=self.stokes.shape
        if self.nx2 != self.nx or self.ny2 != self.ny:
            print('params.idl is {}x{} but database_prof.npy is {}x{}'.format(self.nx,self.ny,self.nx2,self.ny2))
            sys.exit(1)
        self.stokes[:,:,:,1] /= self.Weights[1]
        self.stokes[:,:,:,2] /= self.Weights[2]
        self.stokes[:,:,:,3] /= self.Weights[3]
        
        print("Reshaping Stokes...")
        self.stokes = np.transpose(self.stokes, axes=(0,1,3,2))
        self.stokes = self.stokes.reshape((self.nx*self.ny, 4 * self.nlam))
        
        self.n_training, self.n_spectral = self.stokes.shape

    def __getitem__(self, index):

        out_stokes = self.stokes[index,:]
        out_model = self.phys[index,:]

        noise=1e-3        
        noiseI = noise * np.random.randn(self.nlam)
        noiseQ = noise * np.random.randn(self.nlam) / self.Weights[1]
        noiseU = noise * np.random.randn(self.nlam) / self.Weights[2]
        noiseV = noise * np.random.randn(self.nlam) / self.Weights[3]

        noise = np.hstack([noiseI, noiseQ, noiseU, noiseV])

        out_stokes += noise
        
        return out_stokes.astype('float32'), out_model.astype('float32')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Visp(object):
    def __init__(self, batch_size, validation_split=0.2, gpu=0, smooth=0.05):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")        
        self.smooth = smooth
        
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
        print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size
        self.validation_split = validation_split        
                
        # torch.backends.cudnn.benchmark = True

        # Training/validation datasets        
        kwargs = {'num_workers': 2, 'pin_memory': False} if self.cuda else {}
        self.dataset = Dataset()

        idx = np.random.permutation(self.dataset.n_training)
        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

        print(f"Training sample size : {len(self.train_index)}")
        print(f"Validation sample size : {len(self.validation_index)}")

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.validation_sampler, shuffle=False, **kwargs)

        # Neural model        
        print("Defining neural network...")
        self.model = model.Neural(n_stokes=self.dataset.n_spectral, n_latent=9, n_hidden=300).to(self.device)
        self.model.weights_init()
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.weights = torch.tensor(self.weights.astype('float32')).to(self.device)

    def init_optimize(self, epochs, lr, resume, weight_decay):

        self.lr = lr
        self.weight_decay = weight_decay        
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = epochs

        p = pathlib.Path('trained/')
        p.mkdir(parents=True, exist_ok=True)
        
        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'trained/{0}'.format(current_time)

        # Copy model
        shutil.copyfile(model.__file__, '{0}_model.py'.format(self.out_name))
        shutil.copyfile( __file__, '{0}_trainer.py'.format(self.out_name))
        self.file_mode = 'w'

        f = open('{0}_hyper.dat'.format(self.out_name), 'w')
        f.write('Learning_rate       Weight_decay     \n')
        f.write('{0}    {1}'.format(self.lr, self.weight_decay))
        f.close()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.MSELoss().to(self.device)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.5)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name), self.file_mode)

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            self.train(epoch)
            self.validate(epoch)
            self.scheduler.step()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = min(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))

        trainF.close()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        n = 1
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (stokes, phys) in enumerate(t):
            stokes = stokes.to(self.device)
            phys = phys.to(self.device)
            
            self.optimizer.zero_grad()
            
            out_phys = self.model(stokes)

            # Loss
            loss = torch.mean(self.weights[None,:]*(out_phys-phys)**2)
                                                        
            loss.backward()

            self.optimizer.step()

            loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
            
            tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)

            t.set_postfix(loss=loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)
            
        self.loss.append(loss_avg)

    def validate(self, epoch):
        self.model.eval()
        t = tqdm(self.validation_loader)
        n = 1        
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (stokes, phys) in enumerate(t):
                stokes = stokes.to(self.device)
                phys = phys.to(self.device)
                        
                out_phys = self.model(stokes)
                
                # Loss
                loss = torch.mean(self.weights[None,:]*(out_phys-phys)**2)

                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

                t.set_postfix(loss=loss_avg)
            
        self.loss_val.append(loss_avg)

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=0.0, type=float,
                    metavar='WD', help='Weigth decay')
    parser.add_argument('--val', '--validation_split', default=0.2, type=float,
                    metavar='VD', help='Validation split')
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    
    parsed = vars(parser.parse_args())

    deepvel = Visp(batch_size=256, gpu=parsed['gpu'], smooth=parsed['smooth'], validation_split=parsed['val'])

    deepvel.init_optimize(100, lr=parsed['lr'], resume=parsed['resume'], weight_decay=parsed['wd'])
    deepvel.optimize()

    #deep_pd_network.get_lr(init_value=1e-8, final_value=1e0, beta=0.98)
