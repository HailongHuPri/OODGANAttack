import torch.nn as nn
from .gan_utils import get_gan_model



def get_derivable_generator(gan_model_name, generator_type):

    if generator_type == 'StyleGAN-z':
        return StyleGAN(gan_model_name, 'z')
    else:
        raise Exception('Please indicate valid `generator_type`')




class StyleGAN(nn.Module):
    def __init__(self, gan_model_name, start):
        super(StyleGAN, self).__init__()
        self.stylegan = get_gan_model(gan_model_name)
        self.start = start
        self.init = False

    def input_size(self):
        if self.start == 'z':
            return [(512,)]


    def cuda(self, device=None):
        self.stylegan.net.cuda(device=device)

    def forward(self, latent):
        z = latent[0]
        if self.start == 'z':
            w = self.stylegan.net.mapping(z)   
            w = self.stylegan.net.truncation(w) 
            x = self.stylegan.net.synthesis(w)
            return x
