"""Script to launch a training job.

  Loads dataset and model from training config information, sets up checkpointing and tensorboard callbacks, and starts training.

  Usage:

    $ python train.py /path/to/config.ini --debug --experiment loss_comparison_runs --suffix trial1

  Options:

    --debug(Boolean): Only train for 5 epochs on limited data, and delete temp logs
    --experiment(string): Optional label for a higher-level directory in which to store this run's log directory
    --initmodel(string): Path to model directory for loading initialization weights
    --suffix(string): Optional, arbitrary tag to append to job name
    --verbose(Boolean): Whether to log all training details
"""
import argparse
import atexit
import os
import pickle
from shutil import copyfile, rmtree

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import ReduceLROnPlateau

import fastmri_data_generator, filepaths, losses, models, multicoil_motion_simulator
from interlacer import (layers, utils)
from interlacer import losses as int_losses
from scripts import (load_model_utils, training_config)

gpus = tf.config.experimental.list_physical_devices('GPU')
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True


if gpus:
  # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

# Parse args
parser = argparse.ArgumentParser(
    description='Train a Fourier-domain neural network to correct corrupted k-space data.')
parser.add_argument('config', help='Path to .ini config file.')
parser.add_argument('--experiment', help='Experiment folder name.')
parser.add_argument('--initmodel', help='Path to folder with model with which to initialize weights.')
parser.add_argument(
    '--suffix',
    help='Descriptive suffix appended to job name.')
parser.add_argument(
    '--debug',
    help='Boolean indicating whether to run small-scale training experiment.',
    action='store_true')
parser.add_argument('--verbose', help='Whether to print tf training details to log.', default=True)

# Set up config
args = parser.parse_args()
config_path = args.config
experiment = args.experiment
initmodel = args.initmodel
verbose = args.verbose
suffix = args.suffix
debug = args.debug

exp_config = training_config.TrainingConfig(config_path)
exp_config.read_config()

# Load dataset

base_data_dir = '/vast/kmotion2/users/nmsingh/dev/dl-motion-correction/data/waltham_sim_tf_v1_dropoffsensmap'

train_dir = os.path.join(base_data_dir,'train')
val_dir = os.path.join(base_data_dir,'val')

train_generator= multicoil_motion_simulator.generate_single_saved_motion_examples(train_dir,exp_config.num_epochs,exp_config.batch_size,exp_config.hyp_model,exp_config.output_domain,exp_config.enforce_dc)

val_generator = multicoil_motion_simulator.generate_single_saved_motion_examples(val_dir,exp_config.num_epochs,exp_config.batch_size,exp_config.hyp_model,exp_config.output_domain,exp_config.enforce_dc)

if(exp_config.hyp_model):
    model = models.get_motion_estimation_model((None,None,2*exp_config.num_coils),
            exp_config.nonlinearity,
            exp_config.kernel_size,
            exp_config.num_features,
            exp_config.num_convs,
            exp_config.num_layers,
            exp_config.num_coils,
            exp_config.hyp_model,
            exp_config.output_domain,
            exp_config.enforce_dc)
elif(exp_config.architecture == 'INTERLACER_RESIDUAL'):
    model = models.get_multicoil_interlacer_model((None,None,2*exp_config.num_coils),
            exp_config.nonlinearity,
            exp_config.kernel_size,
            exp_config.num_features,
            exp_config.num_convs,
            exp_config.num_layers,
            exp_config.num_coils,
            exp_config.hyp_model,
            exp_config.output_domain)
elif(exp_config.architecture == 'CONV_RESIDUAL'):
    model = models.get_multicoil_conv_model((None,None,2*exp_config.num_coils),
            exp_config.nonlinearity,
            exp_config.kernel_size,
            exp_config.num_features,
            exp_config.num_layers,
            exp_config.num_coils)
else:
    raise ValueError('This model not supported for multicoil data.')
print('Set up model')

# Checkpointing
job_name = exp_config.job_name

if(debug):
    job_name = 'debug_job' + str(np.random.randint(0, 10))
if(suffix is not None):
    job_name += '*' + suffix
dir_path = filepaths.TRAIN_DIR

if(experiment is not None and not debug):
    dir_path += experiment + '/'
checkpoint_dir = os.path.join(dir_path, job_name)
checkpoint_name = 'cp-{epoch:04d}.ckpt'
checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
cp_callback = keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True, period=10)
print('Set up checkpointing')

if(debug):
    def del_logs():
        rmtree(checkpoint_dir, ignore_errors=True)
        print('Deleted temp debug logs')
    atexit.register(del_logs)

copyfile(args.config, os.path.join(checkpoint_dir, job_name + '_config.ini'))
summary_file = os.path.join(checkpoint_dir, 'summary.txt')
with open(summary_file, 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

# Tensorboard
tb_dir = os.path.join(checkpoint_dir, 'tensorboard/')
if(initmodel is not None):
    pass
elif os.path.exists(tb_dir):
    raise ValueError(
        'Tensorboard logs have already been created under this name. '+ tb_dir)
else:
    os.makedirs(tb_dir)
tb_callback = keras.callbacks.TensorBoard(
    log_dir=tb_dir, histogram_freq=0, write_graph=True, write_images=True)

# Reduce LR on plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=200,
    verbose=2
)

if(not(exp_config.enforce_dc)):
# Select loss
    if(exp_config.loss_type == 'multicoil_ssim'):
        used_loss = losses.multicoil_ssim(exp_config.output_domain,
                exp_config.num_coils, exp_config.enforce_dc)
    elif(exp_config.loss_type == 'multicoil_l1'):
        used_loss = losses.multicoil_l1(exp_config.output_domain,
                exp_config.num_coils)

    # Setup model
    multicoil_ssim = losses.multicoil_ssim(exp_config.output_domain,
            exp_config.num_coils)
    multicoil_l1 = losses.multicoil_l1(exp_config.output_domain,
                exp_config.num_coils)

    metrics = [multicoil_ssim, multicoil_l1]

    lr = 1e-3

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=lr),
        loss=used_loss,
        metrics = metrics)

else:
    lr = 1e-3

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            lr=lr))

print('Compiled model')

if(debug):
    print('Number of parameters: ' + str(model.count_params()))

# Load pre-trained weights, if specified
if(initmodel is not None):
    ckpt_num = load_model_utils.get_best_ckpt(initmodel)
    ckpt = str(ckpt_num).zfill(4)
    ckptname = 'cp-' + ckpt + '.' + 'ckpt'
    model.load_weights(os.path.join(initmodel, ckptname))

# Train model
if(debug):
    num_epochs = 5
    steps_per_epoch = 2
    val_steps = 1
else:
    num_epochs = exp_config.num_epochs
    steps_per_epoch = 10
    val_steps = 8

if(initmodel is not None):
    init_epoch = ckpt_num
else:
    init_epoch = 0

model.fit_generator(
    train_generator,
    epochs=num_epochs,
    steps_per_epoch = steps_per_epoch,
    initial_epoch=init_epoch,
    validation_data=val_generator,
    validation_steps=val_steps,
    callbacks=[
        cp_callback,
        tb_callback,
        reduce_lr],
    verbose=2,
    workers=1)
