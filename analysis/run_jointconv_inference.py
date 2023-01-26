from interlacer import utils
import filepaths, models, losses, multicoil_motion_simulator, diff_forward_model

from random import shuffle
import numpy as np
import os
import tensorflow as tf

# Constants
BASE_DATA_DIR = os.path.join(filepaths.DATA_DIR,'waltham_sim_smallmotions/test')
JOINTCONV_MODEL_DIR = os.path.join(filepaths.TRAIN_DIR,'parammatch_smallmotions/MOCO-44-False-INTERLACER_RESIDUAL-9-48-1-6-SSIM-FREQ-GRAPPA-FREQ-False-False-5000-6/')
JOINTCONV_MODEL_EPOCH = 1200
BASE_HYPERMODEL_DIR = os.path.join(filepaths.TRAIN_DIR,'parammatch_smallmotions/MOCO-44-True-INTERLACER_RESIDUAL-3-32-1-6-SSIM-FREQ-GRAPPA-FREQ-True-False-5000-6/')
INFERENCE_RESULTS_DIR = os.path.join(BASE_HYPERMODEL_DIR, 'opt_TEST-inference_results_SGD_multiplerestarts-ep'+str(1800))+'-recompute_ssim'
N_SHOTS = 6
N_COILS = 44

def get_jointconv_model(model_dir):
    model_str = model_dir.split('/')[2][8:]
    hyp_model = model_str[:4]=='True'       

    model = models.get_motion_estimation_model(
            (None, None, 2*N_COILS),
            9,
            48,
            1,
            N_SHOTS,
            N_COILS,
            hyp_model=hyp_model,
            use_gt_params=hyp_model)

    model.compile()    

    epoch_str = str(JOINTCONV_MODEL_EPOCH).zfill(4)
    model.load_weights(model_dir+'cp-'+epoch_str+'.ckpt')
    return model

def get_jointconv_recon(model,sl):
    test_gen = multicoil_motion_simulator.MoCoDataSequence(BASE_DATA_DIR, 1, hyper_model=True, output_domain='FREQ', enforce_dc=True, use_gt_params=True, input_type='GRAPPA',load_all_inputs=True)   
    inputs, outputs = test_gen.__getitem__(None,sl=sl)

    return model(inputs)['k_pred'], outputs['k_true'], outputs['k_corrupt']


if __name__ == "__main__":
    sls = os.listdir(BASE_DATA_DIR)
    shuffle(sls)
    jointconv_model = get_jointconv_model(JOINTCONV_MODEL_DIR)

    for sl in sls:
        if(sl in os.listdir(INFERENCE_RESULTS_DIR)):
            print(sl)
            jointconv_recon, k_true, k_corrupt = get_jointconv_recon(jointconv_model, sl)

            signal_energy = np.sum(np.abs(utils.join_reim_channels(tf.convert_to_tensor(k_corrupt)))**2)
        
            sl_file = os.path.join(INFERENCE_RESULTS_DIR,sl)
            data = np.load(sl_file)
            data = dict(data)
            data['k_jointconv'] = jointconv_recon
            data['ssim_jointconv'] = losses.multicoil_ssim('FREQ', N_COILS, ignore_border=True)(tf.convert_to_tensor(k_true), tf.convert_to_tensor(jointconv_recon))
            data['signal_energy'] = signal_energy

            np.savez(sl_file,**data)