import os
import csv
import h5py
import pickle
import argparse
import PIL.Image
import numpy as np

def process_cifar10(train_img_dir, test_img_dir, h5py_path):
    """
    Processes CIFAR-10 dataset images, reshapes them, and saves as HDF5 format.

    Parameters:
    - train_img_dir (str): Directory containing the training batch files of CIFAR-10.
    - test_img_dir (str): File path for the test batch file of CIFAR-10.
    - h5py_path (str): Path to save the HDF5 file containing the processed datasets.

    The function loads training and testing data from specified directories, reshapes the images from 
    flattened arrays into 3D arrays (channels, height, width), and saves the datasets into an HDF5 file.
    """

    # Process training data from batch files
    data = []
    targets = []
    for i in range(1,6):
        file_path = os.path.join(train_img_dir, 'data_batch_{}'.format(i))
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])

    train_images = np.vstack(data).reshape(-1, 3, 32, 32)
    train_labels = np.array(targets)
    print(f'Training images shape: {train_images.shape}, Training labels shape: {train_labels.shape}')


    # Process testing data from a single batch file
    data = []
    targets = []

    with open(test_img_dir, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
        data.append(entry['data'])
        if 'labels' in entry:
            targets.extend(entry['labels'])
        else:
            targets.extend(entry['fine_labels'])

    test_images = np.vstack(data).reshape(-1, 3, 32, 32)
    test_labels = np.array(targets)

    print(f'Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}')
    print(f'Max label: {np.max(test_labels)}, Min label: {np.min(test_labels)}')

    # Save the arrays into an HDF5 file with specified paths
    with h5py.File(h5py_path, "w") as f:
        f.create_dataset('train_images', data = train_images, dtype='uint8')
        f.create_dataset('train_labels', data = train_labels, dtype='int32')
        f.create_dataset('test_images', data = test_images, dtype='uint8')
        f.create_dataset('test_labels', data = test_labels, dtype='int32')





def process_gtsrb43(train_img_dir, test_img_dir, h5py_path, num_classes = 43, resolution = 32):
    """
    Processes the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

    Parameters:
    - train_img_dir (str): Directory containing training images sorted into subdirectories by class.
    - test_img_dir (str): Directory containing test images and the CSV file with their labels.
    - h5py_path (str): Path to save the HDF5 file containing the processed datasets.
    - num_classes (int): Number of classes in the dataset.
    - resolution (int): The resolution to which images are resized (both height and width).

    The function reads images from the specified directories, resizes them, converts them to the
    appropriate format, and then saves them along with their labels in an HDF5 file.
    """

    # Collecting paths to all training images
    train_img_files = []
    for i in range(num_classes):
        file_path = os.path.join(train_img_dir, '{}'.format(i))
        fileList = os.listdir(file_path)
        for pic in fileList:
            path = os.path.join(file_path, pic)
            train_img_files.append((path, i))

    # Preparing arrays to hold processed training images and labels
    train_images = np.zeros((len(train_img_files), 3, resolution, resolution), dtype=np.uint8)
    train_labels = np.zeros((len(train_img_files)), dtype=np.int32)
    for i_idx, i_content in enumerate(train_img_files):
        img = PIL.Image.open(i_content[0])
        img = img.resize(size=(resolution, resolution), resample=PIL.Image.BICUBIC) 
        img = np.asarray(img)
        img = img.transpose(2, 0, 1) 
        train_images[i_idx] = img
        train_labels[i_idx] = int(i_content[1])
        if i_idx% 1000 ==0:
            print('Processing training image index:',i_idx,flush=True)

        
    # Collecting paths to all test images using info from the CSV
    test_img_files = []
    csv_path = os.path.join(test_img_dir, 'Test.csv')
    test_img_dir = os.path.join(test_img_dir, 'Test')
    fileList = os.listdir(test_img_dir)
    fileList.remove('GT-final_test.csv')
    fileList.sort(key=lambda x: int(x[: 5]))
    with open(csv_path) as f:
        csv_reader = csv.reader(f)
        t = []
        for row in csv_reader:
            t.append(row)
    cnt = 1
    for pic in fileList:
        path = os.path.join(test_img_dir, pic)
        label = int(t[cnt][6])
        test_img_files.append((path, label))
        cnt = cnt + 1

    # Preparing arrays to hold processed test images and labels
    test_img_files = np.array(test_img_files)
    test_images = np.zeros((test_img_files.shape[0], 3, resolution, resolution), dtype=np.uint8)
    test_labels = np.zeros((test_img_files.shape[0]), dtype=np.int32)

    # Process each test image, resize and store in the array
    for i_idx, i_content in enumerate(test_img_files):
        img = PIL.Image.open(i_content[0])
        img = img.resize(size=(resolution, resolution), resample=PIL.Image.BICUBIC) 
        img = np.asarray(img)
        img = img.transpose(2, 0, 1) 
        test_images[i_idx] = img
        test_labels[i_idx] = int(i_content[1])
        if i_idx% 1000 ==0:
            print('Processing test image index:',i_idx,flush=True)  

    # Save processed images and labels to an HDF5 file
    with h5py.File(h5py_path, "w") as f:
        f.create_dataset('train_images', data = train_images, dtype='uint8')
        f.create_dataset('train_labels', data = train_labels, dtype='int32')
        f.create_dataset('test_images', data = test_images, dtype='uint8')
        f.create_dataset('test_labels', data = test_labels, dtype='int32')               

def main(args):
    """
    Main function to process datasets based on provided arguments.
    
    Args:
        args: Command line arguments containing dataset and path information.
    
    Based on the dataset name specified in command line arguments ('cifar10' or 'gtsrb43'),
    this function will call the appropriate processing function with the paths provided.
    """
    
    if args.data_name == 'cifar10':
        process_cifar10(args.train_img_dir, args.test_img_dir, args.h5py_file)
    elif args.data_name == 'gtsrb43':
        process_gtsrb43(args.train_img_dir, args.test_img_dir, args.h5py_file)

if __name__ == '__main__':
    # Set up command line argument parsing.
    parser = argparse.ArgumentParser(description='Process datasets and store in HDF5 format.')
    parser.add_argument('--train_img_dir', default='./', type=str, help='Path for the training data folder.')
    parser.add_argument('--test_img_dir', default='./', type=str, help='Path for the test data folder.')
    parser.add_argument('--h5py_file', default='./dataset.h5py', type=str, help='Path to save processed data as HDF5.')
    parser.add_argument('--data_name', choices=['cifar10', 'gtsrb43'], default='cifar10', type=str,
                        help='Specify which dataset to process: "cifar10" or "gtsrb43".')
    
    args = parser.parse_args()
    
    # Display the settings before running the main function
    print('Settings:')
    for arg in vars(args):
        print(f'\t{arg}: {getattr(args, arg)}')

    # Execute the main function with the parsed arguments
    main(args)
