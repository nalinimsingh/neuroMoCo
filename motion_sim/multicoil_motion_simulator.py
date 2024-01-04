from random import sample, shuffle
from scipy import ndimage, interpolate
from tqdm import tqdm
import csv
import filepaths
import motion_sim.diff_forward_model as diff_forward_model, motion_sim.nufft_moco as nufft_moco
import multiprocessing
import numpy as np
import os
import tensorflow as tf
from pydicom import read_file
import pygrappa

from interlacer import utils
import motion_sim.diff_forward_model as diff_forward_model
import mrimotion as mot

N_SHOTS = 6

def run_grappa(kspace):
    k_line = kspace[0,:,0]

    # Find the index of the midpoint
    midpoint_index = len(k_line) // 2

    # Find the start index of the contiguous nonzero elements
    acs_start = midpoint_index - np.argmax(k_line[:midpoint_index][::-1] == 0)

    # Find the end index of the contiguous nonzero elements
    acs_end = midpoint_index + np.argmax(k_line[midpoint_index:] == 0)

    grappa = pygrappa.grappa(kspace,kspace[:,acs_start:acs_end,:])

    return grappa


def get_edge_multiplier(shape):
    rows = np.zeros(shape).T
    rows[:] = range(shape[0])
    rows = rows.T

    cols = np.zeros(shape)
    cols[:] = range(shape[1])

    multiplier = np.ones(shape)

    w = 20
    multiplier[:w,:] = rows[:w,:]/w
    multiplier[-w:,:] = (shape[0]-rows[-w:,:])/w

    multiplier[:,:w] = np.minimum(multiplier[:,:w],cols[:,:w]/w)
    multiplier[:,-w:] = np.minimum(multiplier[:,-w:],(shape[1]-cols[:,-w:])/w)

    return multiplier

def extend_float_maps(fmap,mask,bg,x,y):
    mask_x, mask_y = np.where(mask)

    z = [fmap[mask_x[i],mask_y[i]] for i in range(mask_x.shape[0])]

    tck = interpolate.bisplrep(mask_x, mask_y, z, xb=0, xe=mask.shape[0], yb=0, ye=mask.shape[1], kx=3, ky=3)
    bspl = interpolate.bisplev(x, y, tck)

    bspl = bspl*get_edge_multiplier(bspl.shape)

    output = mask*fmap + ~mask*bspl
    return bspl,output


def extend_sens_maps(maps):
    ext_maps = np.zeros(maps.shape,dtype=np.complex64)
    for coil in range(maps.shape[2]):
        mag_map = np.abs(maps[...,coil])
        re_map = np.real(maps[...,coil])
        im_map = np.imag(maps[...,coil])

        mask = mag_map!=0
        bg = mag_map==0

        x = np.arange(maps.shape[0])
        y = np.arange(maps.shape[1])

        re_bspl, re_out = extend_float_maps(re_map,mask,bg,x,y)
        im_bspl, im_out = extend_float_maps(im_map,mask,bg,x,y)

        bspl = re_bspl+1j*im_bspl
        output = re_out+1j*im_out

        ext_maps[...,coil] = bspl

    norm = np.expand_dims(np.sqrt(np.sum(np.square(np.abs(ext_maps)),axis=2)),-1)
    ext_maps = np.divide(ext_maps,norm,out=np.zeros_like(ext_maps),where=norm!=0)

    return ext_maps

def remove_edge(img_recon,maps):
    w = 10

    map_mask = (np.sum(np.abs(maps),axis=2)>0)
    map_mask[:w,:] = 1
    map_mask[-w:,:] = 1
    map_mask = np.expand_dims(map_mask,0)
    masked_recon = np.ma.array(img_recon,mask=map_mask)

    noise_mean = np.mean(masked_recon)
    noise_std = np.std(masked_recon)
    noise = np.random.normal(noise_mean, noise_std, img_recon.shape)

    fill_img_recon = img_recon.copy()
    fill_img_recon[0,:w,:] = noise[0,:w,:]
    fill_img_recon[0,-w:,:] = noise[0,-w:,:]

    return fill_img_recon


def sim_motion(kspace, maps, order_ky, noise_stats=None, max_htrans=0.03, max_vtrans=0.03, max_rot=0.03):
    kspace_shift = np.fft.ifftshift(kspace,axes=(0,1))
    img_slice = np.fft.fftshift(np.fft.ifftn(kspace_shift,axes=(0,1)),axes=(0,1))
    img_recon = np.expand_dims(np.sqrt(np.sum(np.square(np.abs(img_slice)), axis=2)),0)

    img_recon = remove_edge(img_recon,maps)
    img_recon = utils.split_reim(img_recon)[0,...]

    n_x = kspace.shape[1]
    n_y = kspace.shape[0]

    num_motions = 1
    motion_shots = [0]
    motion_shots.extend(np.sort(np.random.randint(1,len(order_ky),size=num_motions)))

    num_points = len(motion_shots)

    num_pix = np.zeros((num_points, 2))
    angle = np.zeros(num_points)

    max_htrans_pix = n_x * max_htrans
    max_vtrans_pix = n_y * max_vtrans
    max_rot_deg = 360 * max_rot

    num_pix[0, :] = np.random.random(2) * (2 * max_htrans_pix) - max_htrans_pix
    num_pix[1:, 0] = np.random.random(
        num_points-1) * (2 * max_htrans_pix) - max_htrans_pix + num_pix[0, 0]
    num_pix[1:, 1] = np.random.random(
        num_points-1) * (2 * max_vtrans_pix) - max_vtrans_pix + num_pix[0, 1]

    angle[0] = np.random.random() - 0.5
    angle[1:] = np.random.random(num_points-1) * (2 * max_rot_deg) - max_rot_deg + angle[0]

    n_shots = 6
    shot_angle = np.zeros((1,n_shots))
    shot_num_pix = np.zeros((1,n_shots,2))

    shot_true = -1
    for shot_i,shot in enumerate(order_ky):
        if(int(kspace.shape[1]/2) in shot):
            shot_true = shot_i
    shot_w_motion = motion_shots[1]

    for i in range(n_shots):
        if(i<shot_w_motion):
            shot_angle[0,i] = angle[0]
            shot_num_pix[0,i,:] = num_pix[0,:]
        else:
            shot_angle[0,i] = angle[1]
            shot_num_pix[0,i,:] = num_pix[1,:]

    shot_order_shape = img_recon.shape[:-1] + (88, 6)
    shot_order_ky = np.zeros(shot_order_shape,dtype=np.complex64)
    for i,shot in enumerate(order_ky):
        shot_order_ky[:,shot,:,i] = 1

    ext_maps = extend_sens_maps(maps)

    k_corrupt, k_true = diff_forward_model.add_rotation_and_translations(img_recon, ext_maps, shot_order_ky, shot_angle[0,:], shot_num_pix[0,...])

    return k_corrupt, k_true, motion_shots, shot_num_pix, shot_angle, ext_maps


def generate_motion_corrupted_brain_data(scan_list_path, maps_dir, write_path):
    np.random.seed(0)
    acq_paths = []
    with open(scan_list_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            acq_paths.append(row[0])

    for acq_path in tqdm(acq_paths):
        acq_data_path = os.path.join(acq_path,'kspace_recon.npz')

        kspace = mot.utils.load_kspace(acq_data_path, 'ge', z_shift=False)
        n_dim = len(kspace.shape)
        if(n_dim!=4):
            continue

        if(kspace.shape[3]<44):
            continue

        n_sl = kspace.shape[2]
        for sl in range(3,n_sl-3):

            for sim_n in range(5):

                _, _, _, _, _, _, subj, scan, _, series = acq_path.split('/')
                sl_str = subj+'_'+scan+'_'+series+'_'+str(sl)+'_sim'+str(sim_n)+'.npz'
                fname = os.path.join(write_path,sl_str)

                if(not sl_str in os.listdir(write_path)):
                    kspace_sl = kspace[:,:,sl,:44]
                    k = kspace_sl[...,np.newaxis].transpose(0,1,3,2)
 
                    # maps = mot.coilsens.run_espirit(k, auto_calib=True)[:,:,0,:]
                    # Read previously computed maps
                    ex_arch = np.load(os.path.join(maps_dir, sl_str), allow_pickle=True)
                    maps = ex_arch['maps'][:,:,0,:]

                    mean, std = 0, 100000

                    file_2d_acqorder = os.path.join(acq_path, 'loopcounters.npz')
                    order_ky, _, _ = mot.acqorder.get_segments(file_2d_acqorder, show_plots=False)
                    order_ky = [order.astype('int32')-1 for order in order_ky]
                    if(len(order_ky)!=N_SHOTS):
                        continue

                    res_info = np.load(os.path.join(acq_path,'resolutioninfo.npz'))
                    psx = res_info['pixelSizeX']
                    psy = res_info['pixelSizeY']

                    k_corrupt, k_true, motion_shots, num_pix, angle, ext_maps = sim_motion(k[:,:,0,:], maps, order_ky, (mean, std))

                    # Add batch dimension
                    k_corrupt = np.expand_dims(k_corrupt, 0)
                    k_true = np.expand_dims(k_true, 0)

                    shot_angle = np.zeros((1,N_SHOTS))
                    shot_num_pix = np.zeros((1,N_SHOTS,2))

                    shot_true = -1
                    for shot_i,shot in enumerate(order_ky):
                        if(int(k_corrupt.shape[2]/2) in shot):
                            shot_true = shot_i

                    shot_w_motion = motion_shots[1]

                    # Compute angles relative to position at central line
                    if(shot_w_motion>shot_true):
                        rel_angle = angle - angle[0,0]
                        rel_num_pix = num_pix - num_pix[0,0,:]
                    else:
                        rel_angle = angle - angle[0,1]
                        rel_num_pix = num_pix - num_pix[0,1,:]

                    shot_order_shape = k_corrupt.shape + (6,)
                    shot_order_ky = np.zeros(shot_order_shape)
                    for shot_i,shot in enumerate(order_ky):
                        shot_order_ky[:,:,shot,:,shot_i] = 1.0

                    mapses = np.expand_dims(ext_maps,0)
                    k_grappa = run_grappa(utils.join_reim_channels(tf.convert_to_tensor(k_corrupt))[0,...])
                    k_grappa = utils.split_reim_channels(tf.expand_dims(k_grappa,0))
                    
                    k_nufft = nufft_moco.nufft_moco(k_grappa, mapses, shot_order_ky, rel_angle, rel_num_pix, use_grappa_interp=True)                    

                    # Write files
                    np.savez(fname,
                             k_corrupt=k_corrupt,
                             k_true = k_true,
                             k_grappa = k_grappa,
                             k_nufft = k_nufft,
                             maps = ext_maps,
                             order_ky = order_ky,
                             shot_order_ky = shot_order_ky,
                             motion_shots = motion_shots,
                             num_pix = num_pix,
                             rel_num_pix = rel_num_pix,
                             angle = angle,
                             rel_angle = rel_angle,
                             psx = psx,
                             psy = psy)


def generate_all_motion_splits():
    base_dir = os.path.join(filepaths.DATA_DIR, 'waltham_sim_smallmotions')
    split_dir = os.path.join(filepaths.DATA_DIR, 'data_splits_rl')
    maps_dir = os.path.join(filepaths.DATA_DIR, 'waltham_sim_v1_multisim')

    for split in ['train','val','test']:
        generate_motion_corrupted_brain_data(os.path.join(split_dir,split+'.csv'),
                os.path.join(maps_dir,split),
                os.path.join(base_dir,split))

class MoCoDataSequence(tf.keras.utils.Sequence):
    def __init__(self, ex_dir, batch_size, hyper_model=False, output_domain='FREQ', enforce_dc=False,
        use_gt_params=False, input_type = 'RAW', load_all_inputs=False):
        self.ex_dir = ex_dir
        self.batch_size = batch_size
        self.hyper_model = hyper_model
        self.output_domain = output_domain
        self.enforce_dc = enforce_dc
        self.use_gt_params = use_gt_params
        self.input_type = input_type
        self.load_all_inputs = load_all_inputs
        
        dir_list = os.listdir(self.ex_dir)
        dir_list = [ex for ex in dir_list if ex[-4:]=='.npz']

        sl_dict = {}
        for ex in dir_list:
            sl = int(ex.split('.')[0].split('_')[3])
            sim = int(ex.split('sim')[1][:-4])
            scan_data = ex.split('.')[0].split('_')[:3]
            sl_files = [ex]
            for _ in range(self.batch_size-1):
                sl = sl+1

                next_sl_str = '_'
                next_sl_str = next_sl_str.join(scan_data)
                next_sl_str += '_'+str(sl)+'_sim'+str(sim)+'.npz'
                sl_files.append(next_sl_str)

            if(all([sl in dir_list for sl in sl_files])):
                sl_dict[ex] = sl_files

        self.sl_dict = sl_dict
        self.sl_dict_inds = list(self.sl_dict.keys())

    def __len__(self):
        return len(self.sl_dict_inds)

    def on_epoch_end(self):
        shuffle(self.sl_dict_inds)

    def __getitem__(self, idx, sl=None):
        if(sl):
            sl_files = [sl]
        else:
            ex = self.sl_dict_inds[idx]
            sl_files = self.sl_dict[ex]
        
        for i,sl in enumerate(sl_files):
            ex_arch = np.load(os.path.join(self.ex_dir, sl), allow_pickle=True)

            if(len(ex_arch['order_ky'])!=N_SHOTS):
                continue

            k_corrupt = ex_arch['k_corrupt']
            norm = np.max(diff_forward_model.rss_image_from_multicoil_k(k_corrupt[0,:,:,:]).numpy().flatten())

            # Initialize appropriately shaped array in the first iteration, for time savings
            if(i == 0):
                k_corrupts = np.empty((self.batch_size,)+k_corrupt.shape[1:], dtype='float32')
                k_trues = np.empty(k_corrupts.shape, dtype='float32')
                k_grappas = np.empty(k_corrupts.shape, dtype='float32')
                k_nuffts = np.empty(k_corrupts.shape, dtype='float32')

                angles = np.empty((self.batch_size,N_SHOTS), dtype='float64')
                num_pixes = np.empty((self.batch_size,N_SHOTS,2), dtype='float64')
                if(self.enforce_dc):
                    order_kys = np.empty(k_corrupts.shape+(N_SHOTS,), dtype='float64')
                    mapses = np.empty(k_corrupts.shape[:-1]+(int(k_corrupts.shape[-1]/2),), dtype='complex64')
                    norms = np.empty((self.batch_size,1), dtype='float32')

            k_corrupts[i,...] = k_corrupt/norm
            k_trues[i,...] = ex_arch['k_true']/norm

            if(self.input_type == 'GRAPPA' or self.load_all_inputs):
                k_grappas[i,...] = ex_arch['k_grappa']/norm
            if(self.input_type == 'NUFFT' or self.load_all_inputs):
                k_nuffts[i,...] = ex_arch['k_nufft']/norm

            if(self.enforce_dc):
                maps = ex_arch['maps']
                mapses[i,...] = np.expand_dims(maps,0)
                norms[i,...] = np.reshape(np.array(norm),(1,1))

                order_ky = ex_arch['order_ky']
                shot_order_shape = k_corrupt.shape + (N_SHOTS,)
                shot_order_ky = np.zeros(shot_order_shape)
                for shot_i,shot in enumerate(order_ky):
                    order_kys[i,:,shot,:,shot_i] = 1.0

            num_pixes[i,...] = ex_arch['rel_num_pix']
            angles[i,...] = ex_arch['rel_angle']

        if(not(np.sum(np.isnan(k_corrupts)) or np.sum(np.isnan(k_trues)))):
            if(self.input_type == 'RAW'):
                k_in = k_corrupts
            elif(self.input_type == 'GRAPPA'):
                k_in = k_grappas
            elif(self.input_type == 'NUFFT'):
                k_in = k_nuffts

            if(self.output_domain=='FREQ'):
                inputs = {'k_in': k_in}

            if(self.use_gt_params):
                inputs['angles'] = angles
                inputs['num_pixes'] = num_pixes

            if(self.load_all_inputs):
                inputs['k_grappa'] = k_grappas
                inputs['k_nufft'] = k_nuffts

            inputs['psx'] = ex_arch['psx']
            inputs['psy'] = ex_arch['psy']

            outputs = {'k_true': k_trues}
            outputs['angle_true'] = angles
            outputs['num_pix_true'] = num_pixes
            outputs['k_corrupt'] = k_corrupts

            if(self.enforce_dc):
                outputs['mapses'] = mapses
                outputs['order_kys'] = order_kys
                outputs['norms'] = norms
                outputs['sl'] = sl

            return(inputs,outputs)

def MoCoDataGenerator(ex_dir, batch_size, hyper_model=False, output_domain='FREQ', enforce_dc=False,
        use_gt_params=False, input_type = 'RAW'):
    seq = MoCoDataSequence(ex_dir, batch_size, hyper_model=hyper_model, output_domain=output_domain, enforce_dc=enforce_dc,
        use_gt_params=use_gt_params, input_type = input_type)

    while True:
        i = np.random.randint(seq.__len__())
        inputs, outputs = seq.__getitem__(i)
        yield (inputs, outputs)


if __name__ == "__main__":
    generate_all_motion_splits()
