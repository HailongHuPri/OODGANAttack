import os
import h5py
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import densenet as dn
import wideresnet as wn


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
  

def checkpoint_best(args, epoch, model, optimizer):
    """
    Saves the best performing model checkpoint to disk.

    Args:
        args (Namespace): Configuration containing model parameters and paths.
        epoch (int): The current epoch number.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer with state to save.

    Description:
        This function checks if the directory for saving the model exists, creates it if not,
        constructs the checkpoint name, and saves the model and optimizer states to a file.
    """    
    save_dir = os.path.join(args.results_dir, 'victim_model_ckpt')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dct = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }
    model_name = args.model_type
    model_name += "-" + args.dataset
    ckpt_name = model_name + '-best.pkl'
    ckpt_path = os.path.join(save_dir, ckpt_name)
    print('Saving checkpoint to {}'.format(ckpt_path))
    torch.save(dct, ckpt_path)
    
        
def adjust_learning_rate(optimizer, epoch, lr_schedule=[50, 75, 90]):
    """
    Adjusts the learning rate based on the epoch according to a pre-defined schedule.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to adjust the learning rate.
        epoch (int): The current epoch count.
        lr_schedule (list of int): Epoch milestones after which the learning rate will decay.

    Description:
        The function reduces the learning rate by a factor of 10 at each scheduled epoch.
        This is typically used to decrease the learning rate in stages, which can lead to
        better overall convergence of the model.
    """
    lr = args.lr
    if epoch >= lr_schedule[0]:
        lr *= 0.1
    if epoch >= lr_schedule[1]:
        lr *= 0.1
    if epoch >= lr_schedule[2]:
        lr *= 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def evaluate(args, loader, model):
    """
    Evaluates the model's performance on a given dataset using the loader.

    Args:
        args (Namespace): Configuration containing model parameters and options.
        loader (DataLoader): DataLoader for the dataset to evaluate.
        model (torch.nn.Module): The model to evaluate.

    Description:
        This function performs evaluation by running the model in inference mode over the dataset.
        It computes the loss and accuracy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    
    with torch.no_grad():
        loss_ts = []
        pred_t = []
        labels = []
        for data_pair in loader:
            # Fit device
            if args.cuda:
                for i in range(len(data_pair)):
                    data_pair[i] = data_pair[i].cuda()
            img = data_pair[0]
            label = data_pair[1]
            logit_t = model(img)
            loss_t = criterion(logit_t, label)
            loss_ts.append(loss_t.item())
            labels += data_pair[1].tolist()
            pred_t += torch.argmax(logit_t, dim=1).tolist()


        top_1 = accuracy_score(labels, pred_t)
        print(f'Top-1 accuracy after evaluation: {top_1:.4f}')
        print('###-------------------------###')
    return top_1


def test_evaluate(args, loader, model):
    """
    Evaluates the model on a test set to measure performance metrics like accuracy and loss.

    Args:
        args (Namespace): Configuration containing model parameters and options.
        loader (DataLoader): DataLoader for the test set.
        model (torch.nn.Module): The model to evaluate.

    Description:
        This function is similar to 'evaluate' but specifically tailored for testing datasets.
        It additionally provides a detailed classification report for better insight into class-wise performance.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    
    with torch.no_grad():
        loss_ts = []
        pred_t = []
        labels = []
        for data_pair in loader:
            # Fit device
            if args.cuda:
                for i in range(len(data_pair)):
                    data_pair[i] = data_pair[i].cuda()

            img = data_pair[0]
            label = data_pair[1]
            logit_t = model(img)
            loss_t = criterion(logit_t, label)
            loss_ts.append(loss_t.item())
            labels += data_pair[1].tolist()
            pred_t += torch.argmax(logit_t, dim=1).tolist()


        print('Classification Report', classification_report(labels, pred_t, digits=4))
        top_1 = accuracy_score(labels, pred_t)
        print(f'Top-1 accuracy after test evaluation: {top_1:.4f}')
        print('###-------------------------###')

def train_target(args, train_loader, val_loader, model, optimizer, lr_schedule):
    """
    Trains the target model with specified training and validation data loaders.

    Args:
        args (Namespace): Configuration containing model parameters and options.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        lr_schedule (list of int): Epochs at which the learning rate should be adjusted.

    Description:
        This function orchestrates the training process which includes setting the model to
        train mode, iterating over the training data, performing backpropagation, and adjusting
        the learning rate. It evaluates the model on the validation set at the end of each epoch
        to monitor performance improvements.
    """    
    criterion = nn.CrossEntropyLoss()
        
    best_top1 = 0.6
    for param in model.parameters():
        param.requires_grad = True    
        
    for epoch in range(args.max_epoch):

        adjust_learning_rate(optimizer, epoch, lr_schedule)

        print('Training epoch {}/{}'.format(epoch, args.max_epoch))
        # Train
        model.train()
        epoch_loss_t = []
        
        for batch_id, data_pair in enumerate(train_loader):
            # Fit device
            if args.cuda:
                for i in range(len(data_pair)):
                    data_pair[i] = data_pair[i].cuda()
            img = data_pair[0]
            label = data_pair[1]
            logit = model(img)
            loss_t = criterion(logit, label)
            loss_t.backward()
            epoch_loss_t.append(loss_t.item())

            optimizer.step()
            optimizer.zero_grad()

  
        print('Training Epoch average loss_t: {}'.format(np.mean(epoch_loss_t)))
        current_top1 = evaluate(args, val_loader, model)
        if current_top1 >best_top1:
            best_top1 = current_top1
            checkpoint_best(args, epoch, model, optimizer)



def main(args):
    
    # Data loader settings
    if args.augment:
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(args.resolution, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  

            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])
        
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    # Dataset-specific settings    
    if args.dataset == 'gtsrb43':
         
        normalizer = None
        lr_schedule=[7, 12, 17]
        num_classes = 43  

        
    elif args.dataset == 'cifar10':
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])        
 
        lr_schedule=[50, 75, 90]
        num_classes = 10

    else:
        raise NotImplementedError("Unknown dataset: %s" % args.dataset)
           
    # Configure data loaders
    if args.model_mode == 'train':
        train_loader = h5py_DataLoader(root=args.data_dir, batch_size=args.batch_size,
                              split='train', transform=transform_train)   
        test_loader = h5py_DataLoader(root=args.data_dir,batch_size=args.batch_size,
                             split='test',transform=transform_test)        
        # Model settings
        if args.model_type == "wideresnet":
            model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, 
                                  dropRate=args.droprate, normalizer=normalizer)

        elif args.model_type == "densenet":

            model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)
        else:
            raise NotImplementedError("Unknown Model name %s" % args.model_tyep)        
        
        # cuda 
        if args.cuda:
            model = model.cuda()        
   
    elif args.model_mode == 'test':
        test_loader = h5py_DataLoader(root=args.data_dir,batch_size=args.batch_size,
                     split='test',transform=transform_test)  
        
        if args.victim_model_type == "wideresnet":
            model = wn.WideResNet(args.depth, num_classes, widen_factor=args.width, 
                                  dropRate=args.droprate, normalizer=normalizer)

            print('Loading victim model checkpoint from {}'.format(args.victim_model_path))
            dct = torch.load(args.victim_model_path)
            model.load_state_dict(dct['model'])
            
            
        elif args.victim_model_type == "densenet":
            model = dn.DenseNet3(args.layers, num_classes, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate, normalizer=normalizer)

            print('Loading victim model checkpoint from {}'.format(args.victim_model_path))
            dct = torch.load(args.victim_model_path)
            model.load_state_dict(dct['model'])         

        if args.cuda:  
            model = model.cuda()     
            
    else:
        raise NotImplementedError("Unknown model_mode: %s" % args.model_mode)
        
        
    if args.optim == "adam":
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.wd)
    elif args.optim == "sgd":
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                              nesterov=True, weight_decay=args.wd)
    else:
        raise NotImplementedError("Unknown Optimizer name %s" % args.optim)


    if args.ckpt_path is not None:
        print('Loading checkpoint from {}'.format(args.ckpt_path))
        dct = torch.load(args.ckpt_path)
        model.load_state_dict(dct['model'])
        optimizer.load_state_dict(dct['optim'])        

   
    # Start     
    if args.model_mode == "train":
        train_target(args, train_loader, test_loader, model, optimizer, lr_schedule)
        _ = evaluate(args, test_loader, model)
       
    elif args.model_mode == 'test' :
        test_evaluate(args, test_loader, model)
    else:
        raise NotImplementedError("Unknown running setting: %s" % args.model_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Hyper-parameters settings
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training and testing')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate for the optimizer')
    parser.add_argument('--wd', default=0.0001, type=float,
                        help='Weight decay to reduce overfitting in the optimizer')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum factor for the optimizer')
    parser.add_argument('--max_epoch', default=100, type=int,
                        help='Maximum number of training epochs')
    parser.add_argument('--optim', default="sgd", type=str,
                        help='Choice of optimizer: sgd')
    parser.add_argument('--resolution', default=32, type=int, 
                        help='Resolution of the input images')
    
    parser.add_argument('--layers', default=100, type=int,
                        help='total number of layers (default: 100)')
    parser.add_argument('--depth', default=40, type=int,
                        help='depth of resnet')
    parser.add_argument('--width', default=4, type=int,
                        help='width of resnet')
    parser.add_argument('--growth', default=12, type=int,
                        help='number of new channels per layer (default: 12)')
    parser.add_argument('--droprate', default=0.0, type=float,
                        help='dropout probability (default: 0.0)')    
    parser.add_argument('--reduce', default=0.5, type=float,
                        help='compression rate in transition stage (default: 0.5)')


    # Load checkpoint
    parser.add_argument('--ckpt_path', default=None, type=str,
                        help='Path to a checkpoint file to load model weights')
    # Run setting
    parser.add_argument('--model_mode', default="", type=str,
                        help="Operational mode: train, test")
    parser.add_argument('--model_type', default="",
                        type=str, help="Model architecture to use: wideresnet, densenet.")
    
    parser.add_argument('--victim_model_path', default="./", type=str,
                        help="Path for loading victim model during tests. Only use in test mode.")
    parser.add_argument('--victim_model_type', default="None",type=str, 
                        help="Type of victim model to load during tests. Only use in test mode.")


    parser.add_argument('--results_dir', default="../dataset",
                        type=str, help="Directory to store results")    
    parser.add_argument('--data_dir', default="../dataset",
                        type=str, help="Directory where dataset files are located")
    parser.add_argument('--dataset', default="cifar10", type=str,
                        help="Dataset to use for training/testing")
    
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument('--bottleneck', dest='bottleneck', action='store_true',default=True,
                    help='To not use bottleneck block')


    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    # Print configuration settings
    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    # Call the main function with parsed arguments
    main(args)


