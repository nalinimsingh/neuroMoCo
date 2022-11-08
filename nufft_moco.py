import numpy as np
import os
import pandas as pd
import tensorflow as tf
from interlacer import utils

import importlib

import torch
import torchkbnufft as tkbn
if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

def nufft_moco(kspace, mapses, order_kys, angles, num_pixes, use_grappa_interp = False):
    kspace_compl = utils.join_reim_channels(tf.convert_to_tensor(kspace))
    trans_kspace = np.zeros(kspace_compl.shape,dtype='complex128')
    nufft_kspace = np.zeros(kspace_compl.shape,dtype='complex128')
    im_size = kspace.shape[1:3]
    
    row_mat = np.repeat(np.expand_dims(np.arange(-int(kspace.shape[1]/2),int(kspace.shape[1]/2))+0.5,-1),kspace.shape[2],axis=1)
    col_mat = np.repeat(np.expand_dims(np.arange(-int(kspace.shape[2]/2),int(kspace.shape[2]/2))+0.5,0),kspace.shape[1],axis=0)
    
    for i in range(kspace.shape[0]):
        fixed_ktraj = []
        k_data = []
        
        if(use_grappa_interp):
            order_kys_to_use = order_kys.copy()
            # Update order_kys to include all lines
            acq_inds = np.nonzero(np.sum(order_kys[i,...,0,:],axis=(0,2)))[0]
            for j in range(order_kys.shape[2]):
                if j in acq_inds:
                    pass
                else:
                    closest_ind = (np.abs(acq_inds - j)).argmin()
                    closest = acq_inds[closest_ind]
                    closest_shot = np.argmax(np.sum(order_kys[i,:,closest,0,:],axis=0))
                    order_kys_to_use[i,:,j,:,closest_shot] = 1
                    
        else:
            order_kys_to_use = order_kys
            
        for shot in range(order_kys.shape[-1]):
            # Apply translations
            row_phase_ramp = np.expand_dims(np.exp(2j*np.pi*row_mat*num_pixes[i,shot,1]/kspace.shape[1]),-1)
            col_phase_ramp = np.expand_dims(np.exp(2j*np.pi*col_mat*num_pixes[i,shot,0]/kspace.shape[2]),-1)

            interm = row_phase_ramp*col_phase_ramp*utils.join_reim_channels(tf.convert_to_tensor(kspace))[i,...]
            trans_kspace[i,...] += order_kys_to_use[i,...,:44,shot]*interm
            
            ktraj_shot = np.nonzero(np.abs(order_kys_to_use[i,...,0,shot]))
            ktraj_shot = np.stack(ktraj_shot,axis=0)

            ktraj_pi = ktraj_shot.copy().astype('float32')
            ktraj_pi[0,...] = (ktraj_shot[0,...]-0.5)/kspace.shape[1]*2*np.pi-np.pi
            ktraj_pi[1,...] = (ktraj_shot[1,...]-0.5)/kspace.shape[2]*2*np.pi-np.pi
            
            # Apply rotations
            theta = -angles[i,shot]*2*np.pi/360
            c, s = np.cos(theta), np.sin(theta)
            rot_mat = np.array(((c, -s), (s, c)))
            
            ktraj_rot = np.matmul(rot_mat,ktraj_pi)
            fixed_ktraj.append(ktraj_rot)
            
            kdata_shot = trans_kspace[i,ktraj_shot[0,...],ktraj_shot[1,...], :]
            k_data.append(kdata_shot)
        
        # Get trajectories
            
        fixed_ktraj = np.concatenate(fixed_ktraj,axis=1)
        k_data = np.concatenate(k_data,axis=0)

        fixed_ktraj = torch.tensor(fixed_ktraj).to(device)
        adjnufft_ob = tkbn.KbNufftAdjoint(
            im_size=im_size,
            grid_size=[i for i in im_size],
        )

        dcomp = tkbn.calc_density_compensation_function(ktraj=fixed_ktraj, im_size=im_size)
        
        
        for coil in range(kspace_compl.shape[3]):
            k_data_coil = np.expand_dims(np.expand_dims(k_data[:,coil],0),0)
            k_data_coil = torch.tensor(k_data_coil).to(device)

            out = adjnufft_ob(k_data_coil * dcomp, fixed_ktraj)
            nufft_kspace[i,...,coil] = out[0,0,...]
            
    nufft_kspace = nufft_kspace/(kspace.shape[1]*kspace.shape[2]/4)
    nufft_kspace = utils.split_reim_channels(tf.convert_to_tensor(nufft_kspace))
    nufft_kspace = tf.signal.fftshift(nufft_kspace,axes=(1,2))
    nufft_kspace = utils.convert_channels_to_frequency_domain(nufft_kspace)
    return nufft_kspace
