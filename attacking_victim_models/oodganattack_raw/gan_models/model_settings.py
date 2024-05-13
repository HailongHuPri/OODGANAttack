# python 3.7
"""Contains basic configurations for models used in this project.

Please download the public released models from the following repositories
OR train your own models, and then put them into the folder
`pretrain/tensorflow`.

StyleGAN: https://github.com/NVlabs/stylegan


NOTE: Any new model should be registered in `MODEL_POOL` before used.
"""

import os

BASE_DIR = os.path.dirname(os.path.relpath(__file__))

MODEL_DIR = 'the path to the pre-trained GAN model'
PTH_MODEL_DIR = 'pytorch'
TF_MODEL_DIR = 'tensorflow'

if not os.path.exists(os.path.join(MODEL_DIR, PTH_MODEL_DIR)):
  os.makedirs(os.path.join(MODEL_DIR, PTH_MODEL_DIR))

# pylint: disable=line-too-long
MODEL_POOL = {
    
    'stylegan_ffhq_32x32': {
        'weight_path': os.path.join(MODEL_DIR, PTH_MODEL_DIR, 'stylegan_ffhq.pth'),
        'tf_weight_path': os.path.join(MODEL_DIR, TF_MODEL_DIR, 'stylegan_ffhq.pkl'),
        'tf_code_path': os.path.join(BASE_DIR, 'stylegan_tf_official'),
        'gan_type': 'stylegan',
        'dataset_name': 'ffhq',
        'z_space_dim': 512,
        'w_space_dim': 512,
        'resolution': 32,
        'fused_scale': 'auto',
    },    

    
     
   
}
# pylint: enable=line-too-long

# Settings for StyleGAN.
STYLEGAN_TRUNCATION_PSI = 0.7  # 1.0 means no truncation
STYLEGAN_TRUNCATION_LAYERS = 8  # 0 means no truncation

STYLEGAN_RANDOMIZE_NOISE = False



# Settings for model running.
USE_CUDA = True

MAX_IMAGES_ON_DEVICE = 4

MAX_IMAGES_ON_RAM = 800
