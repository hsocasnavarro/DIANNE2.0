#!/usr/bin/env python3.7
import numpy as np, random
import torch
import model
import scipy.io as io
import datetime, sys

class Hinode(object):
    npars=9
    def __init__(self, model_checkpoint, gpu=0, nlam=175):
        """
        Class that can be used to quickly invert Stokes parameters from the HINODE/SP instrument

        Parameters
        ----------
        gpu : int, optional
            Index of the GPU to be used, if available. If not, the code fallbacks to CPU
        model_checkpoint : str
            File with the neural network weights
        """

        # Check if a GPU is available
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device("cuda:{}".format(self.gpu) if self.cuda else "cpu")

        # Define neural model
        print("Restoring neural network...")
        self.model = model.Neural(n_stokes=4*nlam, n_latent=self.npars, n_hidden=300).to(self.device)

        # Load weights
        self.model_checkpoint = '{0}'.format(model_checkpoint)
        checkpoint = torch.load(self.model_checkpoint, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])

        for param in self.model.parameters():
            param.requires_grad = False

        print("   => loaded checkpoint for model '{}'".format(self.model_checkpoint))

        # Count number of free weights
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters())))

    def forward(self, stokes):
        """
        Evaluate the model

        Parameters
        ----------
        stokes : float
            Array of size (n_pixels x 4*nlam) containing the four Stokes profiles of n_pixels pixels
        """
        self.model.eval()

        with torch.no_grad():
            stokes = torch.tensor(stokes.astype('float32'))
            stokes = stokes.to(self.device)

            out_phys = self.model(stokes)

        out_phys = out_phys.cpu().numpy()

        return out_phys        


if (__name__ == '__main__'):

    Weights=[1.,1e-2,1e-2,1e-1]

    print("Reading database with Stokes profiles...")
    tmp=io.readsav('./obs.prof.idl')
    [nlam,ny,nx]=tmp['stki'].shape
    stokes=np.zeros((nx,ny,nlam,4))
    stokes[:,:,:,0]=np.transpose(tmp['stki'], axes=(2,1,0) )
    stokes[:,:,:,1]=np.transpose(tmp['stkq'], axes=(2,1,0) )
    stokes[:,:,:,2]=np.transpose(tmp['stku'], axes=(2,1,0) )
    stokes[:,:,:,3]=np.transpose(tmp['stkv'], axes=(2,1,0) )

    print("Normalizing Stokes parameters...")
    stokes[:, :, :, 1] /= Weights[1]
    stokes[:, :, :, 2] /= Weights[2]
    stokes[:, :, :, 3] /= Weights[3]


    print("Reshaping Stokes...")
    stokes = np.transpose(stokes, axes=(0, 1, 3, 2))
    stokes = stokes.reshape((nx*ny, nlam*4))

    # Neural network
    deephinode = Hinode(model_checkpoint='ann.pth', gpu=3, nlam=nlam)

    normalization = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e5])
    normalization_mean = np.array([5e3, 5e3, 5e3, 5e3, 5e3, 0, 0, 0, 0])
    
    # Disambiguation
    ibx=6
    ind = np.where(phys[:,ibx] < 0.0)[0]
    if len(ind) > 0: print("Disambiguating {} profiles".format(len(ind)))
    phys[ind,ibx] = -phys[ind,ibx]
    phys[ind,ibx+1] = -phys[ind,ibx+1]

    #breakpoint()
    start=datetime.datetime.now()
    out_phys = deephinode.forward(stokes[:, :])
    end=datetime.datetime.now()
    print('Inversion done in {} seconds'.format((end-start).total_seconds()))

    # empirical calibration
    calibnorm=[.896,1.265,1.292,1.445,1.732,0.7,0.7,0.7,.7353]

    for i in range(npars):
        out_phys[:,i]=out_phys[:,i]*normalization[i]+normalization_mean[i]
        out_phys[:,i]=out_phys[:,i]+calibdiff[i]
        med=np.median(out_phys[:,i])
        out_phys[:,i]=(out_phys[:,i]-med)*calibnorm[i] + med
        
    images=out_phys.reshape((nx,ny,npars))

    np.savez('images.npz',images=images)
