import os
import h5py
import torch
import argparse
import numpy as np 
import torch.nn as nn


from inversion.losses import get_loss
from inversion.inversion_methods import get_inversion
from utils.image_precossing import _tanh_to_sigmoid
from derivable_models.derivable_generator import get_derivable_generator


from victim_models import densenet as dn
from victim_models import wideresnet as wn


import torch.utils.data as data
import torchvision.transforms as transforms


import random
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class h5py_Data(data.Dataset):
    """
    A custom dataset class that interfaces with HDF5 files using the h5py library.
    It supports loading training and testing data from specified HDF5 files.
    """
    def __init__(self, root, split='train', transform=None):
        """
        Initializes the dataset class with file path, data split, and transformations.

        Args:
            root (str): The file path of the HDF5 file.
            split (str, optional): The type of dataset split to load ('train' or 'test'). Defaults to 'train'.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.
        """
        super().__init__()
        self.transform = transform
        if split == 'train':            
            with h5py.File(root, "r") as hf:
                self.data = hf['train_images'][:] 
                self.labels = hf['train_labels'][:]

        elif split == 'test':
            with h5py.File(root, "r") as hf:
                self.data = hf['test_images'][:]
                self.labels = hf['test_labels'][:]
                
        else:
            raise NotImplementedError
        

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves an item by its index.

        Args:
            index (int): Index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        img = self.data[index]
        img = img.transpose(1, 2, 0) 
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
def h5py_DataLoader(root, batch_size=64, num_workers=2, split='train', transform=None):
    """
    Creates a data loader for the custom dataset.

    Args:
        root (str): The file path of the HDF5 file.
        batch_size (int, optional): Number of samples in each batch. Defaults to 64.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 2.
        split (str, optional): The type of dataset split to load ('train' or 'test'). Defaults to 'train'.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Defaults to None.

    Returns:
        DataLoader: A DataLoader object configured with the specified dataset.
    """    
    dataset = h5py_Data(root, split, transform) 
    if split == 'train':
        shuffle=True
    else:
        shuffle=False
        
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers)



def choose_ref_samples(args, victim_model, num_classes,repeat_num):
    print('----prepare reference samples')
    transform = transforms.Compose([transforms.ToTensor(),
                                    ])        
    ref_loader = h5py_DataLoader(root=args.victim_images_path,batch_size=1,
                   split='test',transform=transform)  

    ref_predicts = [] 
    real_labels = []
    confidences = []
    preset_conf = 0.5
    conf_probs = nn.Softmax(dim=1)

    # No gradient computation to speed up the process
    with torch.no_grad():
        for i_batch, ref_data in enumerate(ref_loader):
            real_labels.append(ref_data[1])
            img = ref_data[0].cuda() 
            predict_t = victim_model(img)    
            
            confidence = conf_probs(predict_t)
            confidences += torch.max(confidence, dim=1)[0].cpu().detach().tolist() 
            ref_predicts.append(predict_t.flatten())

        # Stack all the predictions for further processing
        ref_predicts = torch.stack(ref_predicts)
        ref_predicts_labels = torch.argmax(ref_predicts, dim=1).cpu().numpy()  
        real_labels = np.array(real_labels)

    # Create a dictionary to classify predictions by class based on confidence and accuracy    
    class_dict = {}
    for i_d in range(num_classes):
        class_dict[i_d] = []
    for idx, c_idx in enumerate(zip(real_labels, ref_predicts_labels)):
        if c_idx[0] == c_idx[1] and confidences[idx] > preset_conf:
            class_dict[c_idx[0]].append(idx)

    # Select indices for repeat samples    
    sele_idx= []       
    for i_d in range(num_classes):
        class_samples = class_dict[i_d]
        class_samples = np.array(class_samples)
        shff_idx = np.random.RandomState(seed=2021).permutation(len(class_samples))
        sele_idx.append(class_samples[shff_idx[:repeat_num]])
                
    sele_idx = np.concatenate( sele_idx, axis=0 )
    ref_labels = ref_predicts_labels[sele_idx]
    ref_labels = torch.from_numpy(ref_labels).cuda()
    ref_predicts = ref_predicts[sele_idx,:]
    
    # Save selected indices and predictions to an H5PY file
    print('----reference samples finish')  
    tmp_path = f'./ref_samp_selected_idx_{num_classes}.h5py'
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset('ref_predicts', data = ref_predicts.detach().cpu().numpy(), dtype='float32')
        f.create_dataset('ref_labels', data = ref_labels.detach().cpu().numpy(), dtype='int32')

    # Return the reference labels and predictions    
    return ref_labels , ref_predicts



def main(args):

    # Create the results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # victim model settings
    if args.victim_dataset_name == 'gtsrb43':
        num_classes = 43
        normalizer = None
    elif args.victim_dataset_name == 'cifar10':
        num_classes = 10    
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])             
    
    # load pretrained model 
    if args.victim_model == 'wideresnet':
        victim_model = wn.WideResNet(40, num_classes, widen_factor=4, 
                              dropRate=0.0, normalizer=normalizer)        
        print('Loading wideresnet model from {}'.format(args.victim_model_path))
        dct = torch.load(args.victim_model_path)
        victim_model.load_state_dict(dct['model'])

    elif args.victim_model == 'densenet':
        victim_model = dn.DenseNet3(100, num_classes, 12, reduction=0.5,
                         bottleneck=True, dropRate=0.0, normalizer=normalizer)
        print('Loading densenet model from {}'.format(args.victim_model_path))
        dct = torch.load(args.victim_model_path)
        victim_model.load_state_dict(dct['model'])        

    else:
        raise NotImplementedError("Unknown victim_model: %s" % args.victim_model)    
        
    victim_model = victim_model.cuda()
    victim_model.eval()

    # set attack labels
    repeat_num = 10
    if args.loss_type == 'lc':
        ref_labels = np.arange(0, num_classes)
        ref_labels = np.repeat(ref_labels, repeat_num)
        ref_labels = torch.from_numpy(ref_labels).cuda()
    
    elif args.loss_type == 'l1' or args.loss_type == 'l2':
        tmp_path = f'./ref_samp_selected_idx_{num_classes}.h5py'
        if os.path.exists(tmp_path):
            with h5py.File(tmp_path, "r") as hf:
                ref_predicts = hf['ref_predicts'][:]  
                ref_labels = hf['ref_labels'][:]
            ref_predicts = torch.from_numpy(ref_predicts).cuda()
            ref_labels = torch.from_numpy(ref_labels).cuda()
        else:
            ref_labels, ref_predicts = choose_ref_samples(args, victim_model, num_classes,repeat_num)
        

    if args.num_images==0:
        num_images = num_classes*repeat_num
    else:
        num_images = args.num_images
    
    # GAN model settings
    if args.inversion_type == 'StyleGAN-z':
        latent_estimates_list = np.zeros((num_images,512))
    else:
        raise NotImplementedError("Unknown inversion type: %s")
        
    
    reconstruction_images = np.zeros((num_images, 3, args.resolution, args.resolution), dtype = float) # CHW    
    
    
    ### load generator
    generator = get_derivable_generator(args.gan_model, args.inversion_type)
    loss = get_loss(args.loss_type, args) 
    generator.cuda()
    generator.eval()
    loss.cuda()
    inversion = get_inversion(args.optimization, args)
    
    query_cnts = []
    for i_batch in range(0, num_images, args.im_batch_size):
        
        if  args.loss_type == 'lc': 
            ref_preds = None
            ref_labs = ref_labels[i_batch : i_batch + args.im_batch_size]
            # attacking      
            latent_estimates = inversion.invert(generator,victim_model, ref_labs, ref_preds,
                                                             loss, batch_size=args.im_batch_size, video=False)         
        elif args.loss_type == 'l1' or args.loss_type == 'l2':
            ref_labs = ref_labels[i_batch : i_batch + args.im_batch_size]
            ref_preds = ref_predicts[i_batch : i_batch + args.im_batch_size,:]
            # attacking      
            latent_estimates = inversion.invert(generator,victim_model, ref_labs, ref_preds,
                                                             loss, batch_size=args.im_batch_size, video=False)         
            
        if args.optimization == 'powell':
           
            latent_estimates_list[i_batch : i_batch + args.im_batch_size, :]= latent_estimates.reshape(-1,512)
            tmp = [torch.from_numpy(latent_estimates).float().view(-1, 512).cuda()]
            
            reconstruction_images[i_batch : i_batch + args.im_batch_size] = _tanh_to_sigmoid(torch.clamp(generator(tmp), -1, 1)).detach().cpu().numpy()        
            print('query_cnt',inversion.query_cnt)
            query_cnts.append(inversion.query_cnt)
        else:

            latent_estimates_list[i_batch : i_batch + args.im_batch_size, :]= latent_estimates[0].cpu().detach().numpy()
            reconstruction_images[i_batch : i_batch + args.im_batch_size] = _tanh_to_sigmoid(torch.clamp(generator(latent_estimates), -1, 1)).cpu().detach().numpy()
            query_cnts = [0]
        if i_batch %1 ==0:
            print('index', i_batch)


    # # Save results to file
    ref_labels = ref_labels.cpu().detach().numpy()
    query_cnts = np.array(query_cnts)

    reconstruction_images_path = f'{args.results_dir}/{args.attack_type}_{args.victim_model}_{args.victim_dataset_name}_{args.gan_model}_{args.inversion_type}_{args.loss_type}_{args.optimization}.h5py'
        
    with h5py.File(reconstruction_images_path, "w") as f:
        f.create_dataset('images', data = reconstruction_images, dtype='float')
        f.create_dataset('latents', data = latent_estimates_list, dtype='float')
        f.create_dataset('ref_labels', data = ref_labels, dtype='int32')
        f.create_dataset('query_cnts', data = query_cnts, dtype='int32')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OODGANAttack - Out-of-Distribution GAN Attack Framework')
    # Image Path and Saving Path
    parser.add_argument('--victim_images_path', default='./', help='Directory path where victim images are stored.')
    parser.add_argument('--victim_model_path', default='./', help='File path to the victim model.')         
    parser.add_argument('--victim_dataset_name',type=str, default='', help='Name of the dataset used for the victim model.')
    parser.add_argument('--victim_model', default='wideresnet', help='Type of victim model to target.', type=str)
    
    parser.add_argument('--results_dir', default='./', help='Path to save results of the attack.')
    
    parser.add_argument('--num_images', type=int, default=0, help='Number of images')
    parser.add_argument('--resolution', type=int, default=32, help='resolution')    
    
    parser.add_argument('--im_batch_size', type=int, default=1, help='batch size for the dataset')   
     
    # Parameters for GAN Inversion
    parser.add_argument('--inversion_type', default='StyleGAN-z', help='Method used for GAN inversion.')
    parser.add_argument('--loss_type', default='lc', help="Loss function to use: ['lc', 'l1', 'l2']")

    
    # Optimization Parameters
    parser.add_argument('--optimization', default='sgd', help="['sdg', 'adam', 'powell']. Optimization method used.")
    parser.add_argument('--init_type', default='Normal', help="['Zero', 'Normal']. Initialization type for optimization.")    
    parser.add_argument('--lr', default=0.1, help='Learning rate.', type=float)
    parser.add_argument('--iterations', default=5000, help='Number of optimization steps.', type=int)
    
    parser.add_argument('--maxiter', default=10, help='maxiter (for scipy optimizer).', type=int)
    parser.add_argument('--maxfev', default=25000, help='maxfev (for scipy optimizer).', type=int)
    parser.add_argument('--xtol', default=0.1, help='xtol (for scipy optimizer).', type=float)
    parser.add_argument('--ftol', default=0.1, help='maxfev (for scipy optimizer).', type=float)
    
    # Available pretrained GAN models
    parser.add_argument('--gan_model', default='stylegan_ffhq_32x32', help='The name of model used.', type=str)
    
    
    parser.add_argument('--attack_type', default='wb', help='Type of attack: white-box (wb) or black-box (bb).', type=str)   
    parser.add_argument('--early_stopping', default='True', help='Whether to stop early if convergence is reached.', type=str)   
    parser.add_argument('--req_conf', default=0, help='Required confidence score. 0 indicates no requirement.', type=float)    
    

    args, other_args = parser.parse_known_args()
    # Print the settings to ensure they're captured correctly
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    # Run the main function with parsed arguments
    main(args)
  


