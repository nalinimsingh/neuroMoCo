from random import sample, shuffle
from scipy import ndimage
from tqdm import tqdm
import csv
import diff_forward_model
import multiprocessing
import numpy as np
import os
import tensorflow as tf
from pydicom import read_file

from interlacer import utils
import mrimotion as mot

def simulate_multicoil_k(image, maps):
    """
    image: (x,y) (not-shifted)
    maps: (x,y,coils)
    """
    image = np.repeat(image[:, :, np.newaxis], maps.shape[2], axis=2)
    sens_image = image*maps
    shift_sens_image = np.fft.ifftshift(sens_image, axes=(0,1))

    k = np.fft.fftshift(np.fft.fft2(shift_sens_image, axes=(0,1)), axes=(0,1))
    return k

def rss_image_from_multicoil_k(k):
    """
    k: (x,y,coils) (shifted)
    img: (x,y) (not-shifted)
    """
    img_coils = np.fft.ifft2(np.fft.ifftshift(k, axes=(0,1)), axes=(0,1))
    img = np.sqrt(np.sum(np.square(np.abs(img_coils)), axis=2))
    img = np.fft.fftshift(img)
    
    return img


def add_rotation_and_translations(sl, maps, order_ky, motion_shots, angle, num_pix, noise_stats=None):
    """Add k-space rotations and translations to input slice.
    At each line in coord_list in k-space, induce a rotation and translation.
    Args:
      sl(float): Numpy array of shape (x, y) containing input image data
      maps(complex): Numpy array of shape (x,y,coils) containing complex-valued sensitivity maps
      order_ky(int): List of lists of lines acquired in each shot, zero-indexed
      motion_shots(int): List of shots at which motion occur
      angle(float): Numpy array of angles by which to rotate the input image; of shape (num_points)
      num_pix(float): List of horizontal and vertical translations by which to shift the input image; of shape (num_points, 2)
      noise_stats(float): Tuple of (noise_mean, noise_std)
    Returns:
      sl_k_corrupt(float): Motion-corrupted k-space version of the input slice, of shape(n, n)
    """
    n = sl.shape[1]
    sl_k_true = simulate_multicoil_k(sl, maps)
    sl_k_combined = np.zeros(sl_k_true.shape, dtype='complex64')

    for i in range(len(motion_shots)):
        # Get all the shots for which the brain is in this position
        if(i!=len(motion_shots)-1):
            shots = range(motion_shots[i],motion_shots[i+1])
        else:
            shots = range(motion_shots[i],len(order_ky))
        
        # Simulate k-space data for this position
        sl_rotate = ndimage.rotate(sl, angle[i], reshape=False, mode='nearest')

        if(len(num_pix.shape) == 1):
            sl_moved = ndimage.interpolation.shift(
                sl_rotate, [0, num_pix[i]], mode='nearest')
        elif(num_pix.shape[1] == 2):
            sl_moved = ndimage.interpolation.shift(
                sl_rotate, [0, num_pix[i, 0]], mode='nearest')
            sl_moved = ndimage.interpolation.shift(
                sl_moved, [num_pix[i, 1], 0], mode='nearest')

        sl_k_after = simulate_multicoil_k(sl_moved,maps)

        # Optionally add noise to measurements
        if(noise_stats is not None):
            noise_mean, noise_std = noise_stats
            re_noise = np.random.normal(loc = noise_mean, scale = noise_std, size = sl_k_combined.shape)
            im_noise = np.random.normal(loc = noise_mean, scale = noise_std, size = sl_k_combined.shape)
            sl_k_after = sl_k_after + re_noise + 1j*im_noise        
        
        for shot in shots:
            sl_k_combined[:,order_ky[shot],:] = sl_k_after[:,order_ky[shot],:]
            if(int(n/2) in order_ky[shot]):
                sl_k_true = sl_k_after
                
    return sl_k_combined, sl_k_true



def sim_motion(kspace, maps, order_ky, noise_stats=None, max_htrans=0.03, max_vtrans=0.03, max_rot=0.03, zero_motion=False):
    """Sample motion parameters and apply them to given k-space data.
    Args:
      sl(float): Numpy array of shape (x, y) containing input image data
      maps(complex): Numpy array of shape (x,y,coils) containing complex-valued sensitivity maps
      order_ky(list): List of lists of lines acquired in each shot, zero-indexed
      noise_stats(float): Tuple of (noise_mean, noise_std)
      max_htrans(float): Largest horizontal translation that can be applied, as a fraction of image width
      max_vtrans(float): Largest vertical translation that can be applied, as a fraction of image width
      max_rot(float): Maximum rotation that can be applied, as a fraction of 360 degrees
      zero_motion(Boolean): Flag indicating whether to ignore motion simulation
    Returns:
      k_corrupt(complex): Numpy array of shape (x,y,coils) containing complex-valued corrupted k-space data
      k_true(complex): Numpy array of shape (x,y,coils) containing complex-valued ground truth k-space data
      motion_shots(int): List of shots at which motion occur
      angle(float): Numpy array of angles by which to rotate the input image; of shape (num_points)
      num_pix(float): List of horizontal and vertical translations by which to shift the input image; of shape (num_points, 2)
    """
    
    kspace_shift = np.fft.ifftshift(kspace,axes=(0,1))
    img_slice = np.fft.fftshift(np.fft.ifftn(kspace_shift,axes=(0,1)),axes=(0,1))
    img_recon = np.sqrt(np.sum(np.square(np.abs(img_slice)), axis=2))

    n_x = kspace.shape[1]
    n_y = kspace.shape[0]

    num_motions = 1
    motion_shots = [0]
    motion_shots.extend(np.sort(np.random.randint(1,len(order_ky),size=num_motions)))

    num_points = len(motion_shots)

    num_pix = np.zeros((num_points, 2))
    angle = np.zeros(num_points)
    
    if(not zero_motion):
        max_htrans_pix = n_x * max_htrans
        max_vtrans_pix = n_y * max_vtrans
        max_rot_deg = 360 * max_rot

        num_pix[:, 0] = np.random.random(
            num_points) * (2 * max_htrans_pix) - max_htrans_pix
        num_pix[:, 1] = np.random.random(
            num_points) * (2 * max_vtrans_pix) - max_vtrans_pix
        angle = np.random.random(num_points) * \
            (2 * max_rot_deg) - max_rot_deg

        k_corrupt, k_true = add_rotation_and_translations(img_recon, maps, order_ky, motion_shots, angle, num_pix, noise_stats)   
    
    return k_corrupt, k_true, motion_shots, num_pix, angle


def generate_motion_corrupted_brain_data(scan_list_path, write_path):
    """Writes npz files containing corrupted and true examples and motion parameters.
    Args:
      scan_list_path(str): Path to CSV file containing paths of k-space data
      write_path(str): Path to directory in which to write npz files
    """
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
            for sim_n in range(1):
                _, _, _, _, _, _, subj, scan, _, series = acq_path.split('/')
                sl_str = subj+'_'+scan+'_'+series+'_'+str(sl)+'_sim'+str(sim_n)+'.npz'
                fname = os.path.join(write_path,sl_str)

                if(not sl_str in os.listdir(write_path)):                
                    kspace_sl = kspace[:,:,sl,:44]
                    k = kspace_sl[...,np.newaxis].transpose(0,1,3,2)
                    maps = mot.coilsens.run_espirit(k, auto_calib=True)
                    
                    # TODO: More sophisticated noise parameter estimation
                    noise_stats = (0, 100000)

                    file_2d_acqorder = os.path.join(acq_path, 'loopcounters.npz')                    
                    order_ky, _, _ = mot.acqorder.get_segments(file_2d_acqorder, show_plots=False)
                    
                    # Convert to zero-indexed lines
                    order_ky = [order.astype('int32')-1 for order in order_ky]

                    res_info = np.load(os.path.join(acq_path,'resolutioninfo.npz'))
                    psx = res_info['pixelSizeX']
                    psy = res_info['pixelSizeY']

                    k_corrupt, k_true, motion_shots, num_pix, angle = sim_motion(k[:,:,0,:], 
                                                                                 maps[:,:,0,:], 
                                                                                 order_ky, 
                                                                                 noise_stats)

                    # Split real and imaginary
                    k_corrupt = np.concatenate([np.real(k_corrupt), np.imag(k_corrupt)], axis=2)
                    k_true = np.concatenate([np.real(k_true), np.imag(k_true)], axis=2)

                    # Add batch dimension
                    k_corrupt = np.expand_dims(k_corrupt, 0)
                    k_true = np.expand_dims(k_true, 0)

                    # Write files
                    np.savez(fname,
                             k_corrupt=k_corrupt,
                             k_true = k_true,
                             maps = maps,
                             order_ky = order_ky,
                             motion_shots = motion_shots,
                             num_pix = num_pix,
                             angle = angle,
                             psx = psx,
                             psy = psy)


def generate_all_motion_splits():
    base_dir = '/vast/kmotion2/users/nmsingh/dev/dl-motion-correction/data/chelsea_sim_debug'
    split_dir = '/vast/kmotion2/users/nmsingh/dev/dl-motion-correction/data/data_splits_rl-chelsea'

    for split in ['train','val','test']:
        generate_motion_corrupted_brain_data(os.path.join(split_dir,split+'.csv'),
                os.path.join(base_dir,split))
                                   
def generate_single_saved_motion_examples(ex_dir, num_epochs, batch_size, output_domain='FREQ', enforce_dc=False):
    """Python generator that yields input/output pairs for training with fit_generator
    Args:
      ex_dir(str): Directory where motion-simulated .npz files are stored
      num_epochs(int): Number of epochs to run training for
      batch_size(int): Number of contiguous slices to include in each batch
      output_domain(str): 'FREQ' or 'IMAGE'; format of model output (and input)
      enforce_dc(Bool): Whether to provide motion parameters needed for data consistency
    """
    dir_list = os.listdir(ex_dir)
    dir_list = [ex for ex in dir_list]

    i = 0
    while True:
        if(i==len(dir_list)):
            shuffle(dir_list)
            i=0

        ex = dir_list[i]
        i+=1

        if(ex[-4:]=='.npz'):
            # Get adjacent slices to form a batch
            sl = int(ex.split('.')[0].split('_')[3])
            sim = int(ex.split('sim')[1][:-4])
            scan_data = ex.split('.')[0].split('_')[:3]

            sl_files = [ex]

            for _ in range(batch_size-1):
                sl = sl+1

                next_sl_str = '_'
                next_sl_str = next_sl_str.join(scan_data)
                next_sl_str += '_'+str(sl)+'_sim'+str(sim)+'.npz'
                sl_files.append(next_sl_str)

            k_corrupts = []
            k_trues = []
            
            order_kys = []
            angles = []
            num_pixes = []
            mapses = []
            norms = []
            
            # For each slice in the batch, read the data
            if(all([sl in dir_list for sl in sl_files])):
                for sl in sl_files:
                    if sl in dir_list:
                        ex_arch = np.load(os.path.join(ex_dir, sl), allow_pickle=True)
                        
                        if(len(ex_arch['order_ky'])!=6):
                            continue
                            
                        k_corrupt = ex_arch['k_corrupt']
                        k_true = ex_arch['k_true']

                        norm = np.max(rss_image_from_multicoil_k(k_corrupt[0,:,:,:]).flatten())
                        k_corrupt /= norm
                        k_true /= norm
                        norms.append(np.reshape(np.array(norm),(1,1)))

                        k_corrupts.append(k_corrupt)
                        k_trues.append(k_true)

                        angle = ex_arch['angle']
                        num_pix = ex_arch['num_pix']
                        order_ky = ex_arch['order_ky']
                        motion_shots = ex_arch['motion_shots']
                        maps = ex_arch['maps'][:,:,0,:]
                        psx = ex_arch['psx']
                        psy = ex_arch['psy']

                        n_shots = 6
                        shot_angle = np.zeros((1,n_shots))
                        shot_num_pix = np.zeros((1,n_shots,2))

                        shot_true = -1
                        for shot_i,shot in enumerate(order_ky):
                            if(int(k_corrupt.shape[2]/2) in shot):
                                shot_true = shot_i
                        shot_w_motion = motion_shots[1]


                        # Compute angles relative to position at central line
                        if(shot_w_motion>shot_true):
                            rel_angle = angle - angle[0]
                            rel_num_pix = num_pix - num_pix[0,:]
                        else:
                            rel_angle = angle - angle[1]
                            rel_num_pix = num_pix - num_pix[1,:]

                        for shot_i in range(n_shots):
                            if(shot_i<shot_w_motion):
                                shot_angle[0,shot_i] = rel_angle[0]
                                shot_num_pix[0,shot_i,:] = rel_num_pix[0,:]
                            else:
                                shot_angle[0,shot_i] = rel_angle[1]
                                shot_num_pix[0,shot_i,:] = rel_num_pix[1,:]

                        shot_order_shape = k_corrupt.shape + (6,)
                        shot_order_ky = np.zeros(shot_order_shape)
                        for shot_i,shot in enumerate(order_ky):
                            shot_order_ky[:,:,shot,:,shot_i] = 1.0

                        order_kys.append(shot_order_ky)
                        angles.append(shot_angle)
                        num_pixes.append(shot_num_pix)
                        mapses.append(np.expand_dims(maps,0))
                
                # Combine all the data for each slice within the batch
                k_corrupt = np.concatenate(k_corrupts,axis=0)
                k_true = np.concatenate(k_trues,axis=0)

                order_kys = np.concatenate(order_kys,axis=0)
                angles = np.concatenate(angles,axis=0)
                num_pixes = np.concatenate(num_pixes,axis=0)
                mapses = np.concatenate(mapses,axis=0)
                norms = np.concatenate(norms,axis=0)              
                img_corrupt = utils.convert_channels_to_image_domain(tf.convert_to_tensor(k_corrupt))
                img_true = utils.convert_channels_to_image_domain(tf.convert_to_tensor(k_true))
                
                # Output in the appropriate format
                if(not(np.sum(np.isnan(k_corrupt)) or np.sum(np.isnan(k_true)))):
                    if(output_domain=='FREQ'):
                        if(enforce_dc):
                            yield((k_corrupt, mapses, order_kys, angles, num_pixes, norms), k_true)
                        else:
                            yield(k_corrupt, k_true)
                    elif(output_domain=='IMAGE'):
                        if(enforce_dc):
                            yield((img_corrupt, mapses, order_kys, angles, num_pixes, norms), img_true)
                        else:
                            yield(img_corrupt, img_true)     


if __name__ == "__main__":
    generate_all_motion_splits()
