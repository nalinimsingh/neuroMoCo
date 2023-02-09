"""Script to automatically generate config files for training.

After editing the lists under 'Customizable fields' with desired experiment configurations, running this script will create a directory called $exp_name at the path in scripts/filepaths.CONFIG_DIR. This script then populates the directory with all combinations of the specified fields, except for those under 'Excluded configs'.

  Usage:

  $ python make_configs.py

"""

import itertools
import math
import os
import shutil

import numpy as np

import filepaths

exp_name = 'sample_motion_exp'

# Customizable fields
tasks = ['MOCO']
num_coilses = ['44']
data_paths = [os.path.join(filepaths.DATA_DIR,'waltham_sim_mediummotions')]
use_gt_paramses = ['True']
architectures = [
    'INTERLACER_RESIDUAL']
kernel_sizes = ['3']
num_featureses = ['32']
num_convses = ['1']
num_layerses = ['6']
losses = ['SSIM']
input_domains = ['FREQ']
input_types = ['RAW','GRAPPA']
output_domains = ['FREQ']
hyp_models = ['True','False']
motinp_models = ['True','False']
enforce_dcs = ['False']

num_epochses = ['5000']
batch_sizes = ['6']


for task, num_coils, data_path, use_gt_params, architecture, kernel_size, num_features, num_convs, num_layers, loss, input_domain, input_type, output_domain, hyp_model, motinp_model, enforce_dc, num_epochs, batch_size in itertools.product(
        tasks, num_coilses, data_paths, use_gt_paramses, architectures, kernel_sizes, num_featureses, num_convses, num_layerses, losses, input_domains, input_types, output_domains, hyp_models, motinp_models, enforce_dcs, num_epochses, batch_sizes):
    
    base_dir = os.path.join(filepaths.CONFIG_DIR, exp_name)
    
#    if not((hyp_model == 'False' and use_gt_params == 'True') or
#           (hyp_model == 'True' and enforce_dc == 'False')):
    if not(False):
        ini_filename = task
        for name in [
                num_coils,
                use_gt_params,
                architecture,
                kernel_size,
                num_features,
                num_convs,
                num_layers,
                loss,
                input_domain,
                input_type,
                output_domain,
                hyp_model,
                motinp_model,
                enforce_dc,
                num_epochs,
                batch_size]:
            ini_filename += '-' + name
        ini_filename += '.ini'

        dest_file = os.path.join(base_dir, ini_filename)

        print('Writing to: ' + dest_file)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        f = open(dest_file, "w")
        f.write('[DATA]\n')
        f.write('task = ' + task + '\n')
        f.write('num_coils = ' + num_coils + '\n')
        f.write('data_path = ' + data_path + '\n')
        f.write('use_gt_params = ' + use_gt_params + '\n')

        f.write('\n')

        f.write('[MODEL]\n')
        f.write('architecture = ' + architecture + '\n')
        f.write('kernel_size = ' + kernel_size + '\n')
        f.write('num_features = ' + num_features + '\n')
        f.write('num_convs = ' + num_convs + '\n')
        f.write('num_layers = ' + num_layers + '\n')
        f.write('loss = ' + loss + '\n')
        f.write('input_domain = ' + input_domain + '\n')
        f.write('input_type = ' + input_type + '\n')
        f.write('output_domain = ' + output_domain + '\n')
        f.write('hyp_model = ' + hyp_model + '\n')
        f.write('motinp_model = ' + motinp_model + '\n')
        f.write('enforce_dc = ' + enforce_dc + '\n')

        f.write('\n')

        f.write('[TRAINING]\n')
        f.write('num_epochs = ' + num_epochs + '\n')
        f.write('batch_size = ' + batch_size + '\n')
        f.close()
