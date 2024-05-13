import numpy as np

import torch
from gan_models.helper import build_generator



def standard_z_sample(size, depth, device=None):
    '''
    Generate a standard set of random Z as a (size, z_dimension) tensor.
    With the same random seed, it always returns the same z (e.g.,
    the first one is always the same regardless of the size.)
    '''
    # Use numpy RandomState since it can be done deterministically
    # without affecting global state
    rng = np.random.RandomState(None)
    result = torch.from_numpy(rng.standard_normal(size * depth).reshape(size, depth)).float()
    if device is not None:
        result = result.to(device)
    return result


def get_gan_model(model_name):
    """
    :param model_name: Please refer `GAN_MODELS`
    :return: gan_model(nn.Module or nn.Sequential)
    """
    # to deal with pggan for feature blending
    gan = build_generator(model_name)

    if model_name.startswith('style'):
        return gan




