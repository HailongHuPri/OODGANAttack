import h5py
import argparse
import numpy as np


import torch
import torch.nn as nn
import torch.utils.data as data


from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms


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
    

    
class Adv_data(data.Dataset):
    """
    Args:
        root (str): The file path to the dataset file.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
  
    Adv_data (reshaped) dataset.
    input [-1,1] NCHW
    """

    def __init__(self, root, transform=None):
        super().__init__()
        self.transform = transform
        with h5py.File(root, "r") as hf:
            self.data = hf['images'][:]
            self.labels = hf['ref_labels'][:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        img = torch.tensor(img, dtype=torch.float)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    

def Adv_dataLoader(root, batch_size=256, shuffle=False, num_workers=1, transform=None):
    """
    Args:
        root (str): The file path to the dataset file.
        batch_size (int): Number of samples in each batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): How many subprocesses to use for data loading.
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.

    Returns:
        DataLoader: An iterable over the dataset according to the specified batch size and shuffle flag.
    """    
    dataset = Adv_data(root, transform)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def evaluate(args, loader, model):
    """
    Evaluates the model's performance on the provided dataset using the specified data loader.

    Args:
        args (Namespace): Contains runtime-specific parameters, such as debug flags.
        loader (DataLoader): The DataLoader for iterating over the dataset.
        model (nn.Module): The neural network model to evaluate.

    Returns:
        list: A list of predicted labels for each data item in the dataset.
    """
    criterion = nn.CrossEntropyLoss()
    conf_probs = nn.Softmax(dim=1)
    with torch.no_grad():
        loss_ts = []
        pred_t = []
        labels = []
        confidences = []

        for data_pair in loader:
            # Fit device
            for i in range(len(data_pair)):
                data_pair[i] = data_pair[i].cuda()
            img = data_pair[0]
            label = data_pair[1]

            logit_t = model(img)   
            confidence = conf_probs(logit_t)
            confidences += torch.max(confidence, dim=1)[0].cpu().detach().tolist() 
            loss_t = criterion(logit_t, label)
            loss_ts.append(loss_t.item())
            labels += data_pair[1].tolist()
            pred_t += torch.argmax(logit_t, dim=1).tolist()

        if args.e_debug==True:
            print('pred_t', pred_t)
            print('confidences', confidences)
            print('###-------------------------###')
    
    return pred_t



def detailed_eval_bb(all_pred, repeat_num, num_classes, args):
    
    with h5py.File(args.OOD_adv_data_path, "r") as hf:
        query_cnts = hf['query_cnts'][:]     
            
    all_pred = np.array(all_pred)
    
    ref_labels = np.arange(0, num_classes)
    ref_labels = np.repeat(ref_labels, repeat_num)


    all_pred_res = []
    for i in range(len(all_pred)):
        if all_pred[i] == ref_labels[i] and query_cnts[i] <= args.threshold:
            all_pred_res.append(1)
        else:
            all_pred_res.append(0)
    all_pred_res = np.array(all_pred_res).reshape((num_classes, repeat_num))

    class_acc = np.mean(all_pred_res,axis=1)

    print('query_cnts', 'mean', np.mean(query_cnts), 'median', np.median(query_cnts), 'std', np.std(query_cnts))
    #best    
    best_acc = np.max(class_acc)
    worst_acc = np.min(class_acc)
    avg_acc = np.mean(class_acc)
    avg_acc_std = np.std(class_acc)
    print('avg_acc', avg_acc, '----', 'avg_acc_std', avg_acc_std)    
    print('best_acc', best_acc)    
    print('worst_acc', worst_acc)
    
def detailed_eval(all_pred, repeat_num, num_classes):
    all_pred = np.array(all_pred)
    
    ref_labels = np.arange(0, num_classes)
    ref_labels = np.repeat(ref_labels, repeat_num)


    all_pred_res = []
    for i, j in zip(all_pred, ref_labels):
        if i == j:
            all_pred_res.append(1)
        else:
            all_pred_res.append(0)
    all_pred_res = np.array(all_pred_res).reshape((num_classes, repeat_num))

    class_acc = np.mean(all_pred_res,axis=1)
 
    best_acc = np.max(class_acc)
    worst_acc = np.min(class_acc)
    avg_acc = np.mean(class_acc)
    avg_acc_std = np.std(class_acc)
    print('avg_acc', avg_acc, '----', 'avg_acc_std', avg_acc_std)    
    print('best_acc', best_acc)    
    print('worst_acc', worst_acc)
        
    

def eval_attack_performace(args):
    
    #load data    
    adv_loader = Adv_dataLoader(root=args.OOD_adv_data_path, 
                              batch_size=args.batch_size, 
                              num_workers = args.num_workers,
                              shuffle=args.shuffle,
                              transform=None)       
    
    # load weights
    if args.victim_dataset_name == 'gtsrb43':
        num_classes = 43
        normalizer = None
    elif args.victim_dataset_name == 'cifar10':
        num_classes = 10    
        normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x/255.0 for x in [63.0, 62.1, 66.7]])      

        
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
        
    
    if args.attack_type == 'wb':    
        all_pred = evaluate(args, adv_loader, victim_model)
        repeat_num = 10
        detailed_eval(all_pred, repeat_num, num_classes)
    elif args.attack_type == 'bb':              
        all_pred = evaluate(args, adv_loader, victim_model)
        repeat_num = 10
        detailed_eval_bb(all_pred, repeat_num, num_classes, args)        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating attack performance')
    # Image Path and Saving Path
    parser.add_argument('--OOD_adv_data_path', default='.',help='Path to the out-of-distribution adversarial data.')
    
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers for data loading.')   
    parser.add_argument('--batch_size', default=1, type=int, help='Number of samples per batch to load.')   
    parser.add_argument("--shuffle", action="store_true", default=False,help='Whether to shuffle the dataset.')
    parser.add_argument("--e_debug", action="store_true", default=False,help='Enable debug mode for more verbose output.')
    parser.add_argument('--threshold', default=25000, type=float, help='Threshold or query counts for decision making in attacks.')   
    
    
    parser.add_argument('--victim_model', default="", type=str, help='Type of victim model to target.')   
    parser.add_argument('--victim_model_path', default="", type=str, help='File path to the victim model.') 
    parser.add_argument('--victim_dataset_name', default="cifar10", type=str, help='Name of the dataset used for the victim model. in-distribution dataset')  
    parser.add_argument('--attack_type', default="wb", type=str, help='Type of attack: white-box (wb) or black-box (bb).')   

    args, other_args = parser.parse_known_args()

    print('Settings:')
    for arg in vars(args):
        print('\t{}: {}'.format(arg, getattr(args, arg)))
    eval_attack_performace(args)



