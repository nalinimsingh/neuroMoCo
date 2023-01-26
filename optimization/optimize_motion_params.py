from interlacer import utils
import filepaths
import training.models as models, training.losses as losses, motion_sim.multicoil_motion_simulator as multicoil_motion_simulator, motion_sim.diff_forward_model as diff_forward_model

import random

from tensorflow import keras
from tensorflow.keras.layers import *
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa

# Constants
BASE_DATA_DIR = os.path.join(filepaths.DATA_DIR,'waltham_sim_smallmotions/test')
BASE_HYPERMODEL_DIR = os.path.join(filepaths.TRAIN_DIR,'parammatch_smallmotions/MOCO-44-True-INTERLACER_RESIDUAL-3-32-1-6-SSIM-FREQ-GRAPPA-FREQ-True-False-5000-6/')
BASE_HYPERMODEL_EPOCH = 1800
INFERENCE_RESULTS_DIR = os.path.join(BASE_HYPERMODEL_DIR, 'opt_TEST-inference_results_SGD_multiplerestarts-ep'+str(BASE_HYPERMODEL_EPOCH))
N_SHOTS = 6
N_COILS = 44

def get_dc_loss(k, k_corrupt, mapses, order_kys, angles, num_pixes, norms):
    imgs = tf.map_fn(diff_forward_model.rss_image_from_multicoil_k, k)[..., 0]
    elems = [utils.split_reim_tensor(imgs),tf.convert_to_tensor(mapses),tf.convert_to_tensor(order_kys,dtype='float32'),angles,num_pixes,norms]
    map_dfm = (lambda x: diff_forward_model.add_rotation_and_translations(x[0], x[1], x[2], x[3], x[4], x[5]))

    forward, _ = tf.map_fn(map_dfm, elems, fn_output_signature=(tf.float32, tf.float32))
    dc_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(utils.join_reim_channels(forward) - utils.join_reim_channels(tf.convert_to_tensor(k_corrupt)))),axis=(1,2,3)),axis=0)

    return forward, dc_loss

def get_np_ssim_loss(k_true, k_pred):
    return(losses.multicoil_ssim('FREQ', N_COILS)(tf.convert_to_tensor(k_true), tf.convert_to_tensor(k_pred)))


def opt_model(h_model):
    # Freeze hypermodel weights
    for l in h_model.layers:
        l.trainable = False

    k_input = keras.layers.Input((None, None, N_COILS*2))  
    ones = keras.layers.Input((1))

    # Add motion params as weights of Dense layers multiplied with 1
    angles_list = [Dense(1, use_bias=False)(ones) for _ in range(N_SHOTS)]
    num_pixes_list = [Reshape((1,2))(Dense(2, use_bias=False)(ones)) for _ in range(N_SHOTS)]

    angles_est = tf.concat(angles_list, axis=1)
    num_pixes_est = tf.concat(num_pixes_list, axis=1)
    
    output = h_model({'k_in': k_input, 'angles': angles_est, 'num_pixes': num_pixes_est}, training=False)['k_pred']

    inputs = ({'k_in': k_input, 'ones': ones})
    outputs = ({'k_pred': output, 'angles': angles_est, 'num_pixes': num_pixes_est})

    return CustomOptModel(inputs = inputs, outputs=outputs)

class CustomOptModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super(CustomOptModel, self).__init__(*args, **kwargs)
    
    #@tf.function
    def compute_losses(self, data, training):        
        inputs, outputs = data
        
        k_true = outputs['k_true']
        angle_true = outputs['angle_true']
        num_pix_true = outputs['num_pix_true']
        k_corrupt = outputs['k_corrupt']
        mapses = outputs['mapses']
        order_kys = outputs['order_kys']
        norms = outputs['norms']
        
        mapses.set_shape((None,None,None,N_COILS))
        k_corrupt.set_shape((None,None,None,N_COILS*2))

        model_outputs = self(inputs, training=training)      
        
        k_pred = model_outputs['k_pred']
        ssim_loss = tf.reduce_mean(losses.multicoil_ssim('FREQ', N_COILS)(k_true, k_pred))

        angle_pred = model_outputs['angles']
        num_pix_pred = model_outputs['num_pixes']

        forward, dc_loss = get_dc_loss(k_pred, k_corrupt, mapses, order_kys, angle_pred, num_pix_pred, norms)

        corr_signal_power = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(utils.join_reim_channels(k_corrupt))),axis=(1,2,3)),axis=0)
        dc_loss_frac = dc_loss/corr_signal_power
        
        rot_loss = tf.reduce_mean(tf.abs(angle_pred - angle_true), axis=(0, 1))
        trans_loss = tf.reduce_mean(tf.abs(num_pix_pred - num_pix_true),axis=(0,1,2))

        rot_loss_first4 = tf.reduce_mean(tf.abs(angle_pred - angle_true)[:,:4], axis=(0, 1))
        trans_loss_first4 = tf.reduce_mean(tf.abs(num_pix_pred - num_pix_true)[:,:4,:],axis=(0,1,2))

        rot_loss_last2 = tf.reduce_mean(tf.abs(angle_pred - angle_true)[:,4:], axis=(0, 1))
        trans_loss_last2 = tf.reduce_mean(tf.abs(num_pix_pred - num_pix_true)[:,4:,:],axis=(0,1,2))

        return ssim_loss, k_pred, angle_pred, num_pix_pred, dc_loss, dc_loss_frac, rot_loss, rot_loss_first4, rot_loss_last2, trans_loss, trans_loss_first4, trans_loss_last2

    #@tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            ssim_loss, k_pred, angle_pred, num_pix_pred, dc_loss, dc_loss_frac, rot_loss, rot_loss_first4, rot_loss_last2, trans_loss, trans_loss_first4, trans_loss_last2 = self.compute_losses(data, training=True)
            loss = dc_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))            

        return {
            "loss": loss,
            "image_ssim_multicoil": ssim_loss,
            "k_pred": k_pred,
            "dc": dc_loss,
            "dc_frac_loss": dc_loss_frac,
            "angle_pred": angle_pred,
            "num_pix_pred": num_pix_pred,
            "rot": rot_loss,
            "rot_first4": rot_loss_first4,
            "rot_last2": rot_loss_last2,
            "trans": trans_loss,
            "trans_first4": trans_loss_first4,
            "trans_last2": trans_loss_last2,
        }


def img_generator(inputs, outputs):
    for _ in range(10000):
        yield(inputs, outputs)


def get_base_hypermodel(dir):
    model_str = dir.split('/')[2][8:]
    hyp_model = model_str[:4]=='True'       

    model = models.get_motion_estimation_model(
            (None, None, 2*N_COILS),
            3,
            32,
            1,
            N_SHOTS,
            N_COILS,
            hyp_model=hyp_model,
            use_gt_params=hyp_model)

    model.compile()    

    epoch_str = str(BASE_HYPERMODEL_EPOCH).zfill(4)
    model.load_weights(dir+'cp-'+epoch_str+'.ckpt')
    return model

def optimize_example(test_gen, sl, model, epochs=80):
    inputs, outputs = test_gen.__getitem__(None,sl=sl)
    k_grappa = inputs['k_grappa']
    k_nufft = inputs['k_nufft']

    k_true = outputs['k_true']
    k_corrupt = outputs['k_corrupt']
    mapses = outputs['mapses']
    order_kys = outputs['order_kys']
    norms = outputs['norms']

    angles_true = outputs['angle_true']
    num_pixes_true = outputs['num_pix_true']

    inputs['k_in'] = k_grappa
    inputs['ones'] = np.ones((1,1))
        
    backward_hnet = opt_model(model)

    clr_s5 = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=1e-7,
        maximal_learning_rate=1e-6,
        scale_fn=lambda x: 1/(2.**(x-1)),
        step_size=5)

    all_optimizers = [tf.keras.optimizers.SGD(clr_s5) for _ in range(4)]
    
    hist = []
    for opt_i in range(len(all_optimizers)):
        for i in range(1,7):
            backward_hnet.layers[i].set_weights(np.zeros((1,1,2)))
            backward_hnet.layers[i+6].set_weights(np.zeros((1,1,1)))
            if(i==2):
                backward_hnet.layers[i].trainable = False
                backward_hnet.layers[i+6].trainable = False
        backward_hnet.compile(all_optimizers[opt_i])
        hist.append(backward_hnet.fit(img_generator(inputs, outputs),epochs=int(epochs), steps_per_epoch=1))

    recons = {}

    recons['Corrupt'] = k_corrupt
    recons['GT'] = k_true
    recons['ARC'] = k_grappa
    recons['Model-Based'] = k_nufft

    opt_epoch_dcs = [hist[i].history["dc"][-1] for i in range(len(all_optimizers))]
    opt_optimizer = np.argmin(opt_epoch_dcs)
    opt_epoch = np.argmin(hist[opt_optimizer].history["dc"])
    angles_pred = hist[opt_optimizer].history["angle_pred"][opt_epoch]
    num_pixes_pred = hist[opt_optimizer].history["num_pix_pred"][opt_epoch]
    recons['Hypernet-Optimized'] = hist[opt_optimizer].history["k_pred"][opt_epoch]

    zero_mot_inputs = inputs.copy()
    zero_mot_inputs['angles'] = np.zeros(inputs['angles'].shape)
    zero_mot_inputs['num_pixes'] = np.zeros(inputs['num_pixes'].shape)
    zero_outputs = model(zero_mot_inputs)['k_pred']
    recons['Hypernet-Zero'] = zero_outputs

    gt_outputs = model(inputs)['k_pred']
    recons['Hypernet-GT'] = gt_outputs

    dc_losses = {}
    dc_losses['Hypernet-Optimized'] = hist[opt_optimizer].history['dc'][-1]
    dc_losses['GT'] = get_dc_loss(k_true, k_corrupt, mapses, order_kys, inputs['angles'], inputs['num_pixes'], norms)[1]
    dc_losses['Hypernet-GT'] = get_dc_loss(gt_outputs, k_corrupt, mapses, order_kys, inputs['angles'], inputs['num_pixes'], norms)[1]
    dc_losses['Model-Based'] = get_dc_loss(k_nufft, k_corrupt, mapses, order_kys, inputs['angles'], inputs['num_pixes'], norms)[1]

    ssim_results = {}
    ssim_results['Corrupt'] = get_np_ssim_loss(k_true, k_corrupt, ignore_border=True)
    ssim_results['ARC'] = get_np_ssim_loss(k_true, k_grappa, ignore_border=True)
    ssim_results['Model-Based'] = get_np_ssim_loss(k_true, k_nufft, ignore_border=True)
    ssim_results['Hypernet-Optimized'] = get_np_ssim_loss(k_true, recons['Hypernet-Optimized'], ignore_border=True)
    ssim_results['Hypernet-Zero'] = get_np_ssim_loss(k_true, zero_outputs, ignore_border=True)
    ssim_results['Hypernet-GT'] = get_np_ssim_loss(k_true, gt_outputs, ignore_border=True)

    return backward_hnet, hist, opt_optimizer, recons, dc_losses, ssim_results, angles_true, angles_pred, num_pixes_true, num_pixes_pred, inputs['psx'], inputs['psy']

if __name__ == "__main__":
    if not(os.path.exists(INFERENCE_RESULTS_DIR)):
        os.makedirs(INFERENCE_RESULTS_DIR)

    sls = os.listdir(BASE_DATA_DIR)
    random.Random(0).shuffle(sls)
    model = get_base_hypermodel(BASE_HYPERMODEL_DIR)

    np.random.seed(0)
    random.seed(0)
    tf.random.set_seed(0)

    for sl in sls[:100]:
        tf.keras.backend.set_learning_phase(0)
        if(not sl in os.listdir(INFERENCE_RESULTS_DIR)):
            test_gen = multicoil_motion_simulator.MoCoDataSequence(BASE_DATA_DIR, 1, hyper_model=True, output_domain='FREQ', enforce_dc=True, use_gt_params=True, input_type='NUFFT',load_all_inputs=True)      
            _, hist, opt_optimizer, recons, dc_losses, ssim_results, angles_true, angles_pred, num_pixes_true, num_pixes_pred, psx, psy = optimize_example(test_gen, sl, model, epochs=80)
            rot_err_end = hist[opt_optimizer].history['rot'][-1]
            rot_err_start = hist[opt_optimizer].history['rot'][0]
            rot_err_end_first4 = hist[opt_optimizer].history['rot_first4'][-1]
            rot_err_start_first4 = hist[opt_optimizer].history['rot_first4'][0]
            rot_err_end_last2 = hist[opt_optimizer].history['rot_last2'][-1]
            rot_err_start_last2 = hist[opt_optimizer].history['rot_last2'][0]

            trans_err_end = hist[opt_optimizer].history['trans'][-1]
            trans_err_start = hist[opt_optimizer].history['trans'][0]
            trans_err_end_first4 = hist[opt_optimizer].history['trans_first4'][-1]
            trans_err_start_first4 = hist[opt_optimizer].history['trans_first4'][0]
            trans_err_end_last2 = hist[opt_optimizer].history['trans_last2'][-1]
            trans_err_start_last2 = hist[opt_optimizer].history['trans_last2'][0]

            sl_file = os.path.join(INFERENCE_RESULTS_DIR,sl)
            np.savez(sl_file, k_out = recons['GT'],
                            k_corrupt = recons['Corrupt'],
                            k_arc = recons['ARC'],
                            k_nufft = recons['Model-Based'],
                            k_hyperopt = recons['Hypernet-Optimized'],
                            k_hyperzero = recons['Hypernet-Zero'],
                            k_hypergt = recons['Hypernet-GT'],
                            dc_opt = dc_losses['Hypernet-Optimized'],
                            dc_gt = dc_losses['GT'],
                            dc_gthyper = dc_losses['Hypernet-GT'],
                            ssim_corrupt = ssim_results['Corrupt'],
                            ssim_arc = ssim_results['ARC'],
                            ssim_model = ssim_results['Model-Based'],
                            ssim_opt = ssim_results['Hypernet-Optimized'],
                            ssim_zero = ssim_results['Hypernet-Zero'],
                            ssim_gthyper = ssim_results['Hypernet-GT'],
                            rot_err_end = rot_err_end,
                            rot_err_start = rot_err_start,
                            rot_err_end_first4 = rot_err_end_first4,
                            rot_err_end_last2 = rot_err_end_last2,
                            rot_err_start_first4 = rot_err_start_first4,
                            rot_err_start_last2 = rot_err_start_last2,                            
                            trans_err_end = trans_err_end,
                            trans_err_start = trans_err_start,
                            trans_err_end_first4 = trans_err_end_first4,
                            trans_err_end_last2 = trans_err_end_last2,
                            trans_err_start_first4 = trans_err_start_first4,
                            trans_err_start_last2 = trans_err_start_last2,  
                            angles_true = angles_true,
                            angles_pred = angles_pred,
                            num_pixes_true = num_pixes_true,
                            num_pixes_pred = num_pixes_pred,
                            psx = psx,
                            psy = psy
                            )

