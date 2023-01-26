import os

# Base directory
BASE_DIR = '' # Insert base directory path

# Directory where simulations are written
DATA_DIR = os.path.join(BASE_DIR,'data')

# Directory where config files are written
CONFIG_DIR = os.path.join(BASE_DIR,'configs')

# Directory where training logs are run
TRAIN_DIR = os.path.join(BASE_DIR, 'training_logs')
