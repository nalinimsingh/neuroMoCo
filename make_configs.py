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

exp_name = 'tf_waltham_sim_v1_testDC'

# Customizable fields
num_coilses = ['44']
architectures = [
    'INTERLACER_RESIDUAL']
kernel_sizes = ['9']
num_featureses = ['64']
num_convses = ['1']
num_layerses = ['6']
loss_types = ['multicoil_ssim']
input_domains = ['FREQ']
output_domains = ['FREQ']
nonlinearities = ['3-piece']
hyp_models = ['True', 'False']
enforce_dcs = ['True', 'False']

num_epochses = ['5000']
batch_sizes = ['6']


for num_coils, architecture, kernel_size, num_features, num_convs, num_layers, loss_type, input_domain, output_domain, nonlinearity, hyp_model, enforce_dc, num_epochs, batch_size in itertools.product(
        num_coilses, architectures, kernel_sizes, num_featureses, num_convses, num_layerses, loss_types, input_domains, output_domains, nonlinearities, hyp_models, enforce_dcs, num_epochses, batch_sizes):
    base_dir = os.path.join(filepaths.CONFIG_DIR, exp_name)

    ini_filename = num_coils
    for name in [
            architecture,
            kernel_size,
            num_features,
            num_convs,
            num_layers,
            loss_type,
            input_domain,
            output_domain,
            nonlinearity,
            hyp_model,
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
    f.write('num_coils = ' + num_coils + '\n')

    f.write('\n')

    f.write('[MODEL]\n')
    f.write('architecture = ' + architecture + '\n')
    f.write('kernel_size = ' + kernel_size + '\n')
    f.write('num_features = ' + num_features + '\n')
    f.write('num_convs = ' + num_convs + '\n')
    f.write('num_layers = ' + num_layers + '\n')
    f.write('loss_type = ' + loss_type + '\n')
    f.write('input_domain = ' + input_domain + '\n')
    f.write('output_domain = ' + output_domain + '\n')
    f.write('nonlinearity = ' + nonlinearity + '\n')
    f.write('hyp_model = ' + hyp_model + '\n')
    f.write('enforce_dc = ' + enforce_dc + '\n')

    f.write('\n')

    f.write('[TRAINING]\n')
    f.write('num_epochs = ' + num_epochs + '\n')
    f.write('batch_size = ' + batch_size + '\n')
    f.close()
