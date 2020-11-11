
import numpy as np 
import cv2 
import yaml 
import pandas as pd 



def make_config(config_path='config.yaml'):
    """
    Helper function to turn a .yaml config file into a dictionary.
    """
    with open(config_path, 'r') as f:
        params = yaml.load(f.read(), Loader=yaml.FullLoader)
    return params

def pytorch_unnormalize(image_batch, mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225]):
    '''
    Takes in a PyTorch batch that has been normalized and unnormalizes it.
    https://discuss.pytorch.org/t/simple-way-to-inverse-normalize-a-batch-of-input-variable/12385
    '''
    image_unnorm = image_batch.new(*image_batch.size())
    image_unnorm[:,0,:,:] = image_batch[:, 0, :, :] * std[0] + mean[0]
    image_unnorm[:,1,:,:] = image_batch[:, 1, :, :] * std[1] + mean[1]
    image_unnorm[:,2,:,:] = image_batch[:, 2, :, :] * std[2] + mean[2]
    return image_unnorm

def viz_to_df(X, y):
    '''
    Helper function to convert a 2-D matrix (X) with labels (y) into a DataFrame 
    for visualization. Helpful for plotting PCA/Isomap projections with Seaborn.
    '''
    df = pd.DataFrame({'z1':X[:,0], 'z2':X[:,1], 'label':y})
    return df

def binarize(y, target):
    """
    Change multiclass to binary
    """
    y = np.where(y == target, True, False)
    return y

def downsample_dataset(images, labels, n, vectorize=False):
    """
    Take images, and labels, and shrink the size of the dataset to n.
    """
    n_= n # the desired sample size
    labels_ = labels # our input for labels
    images_ = images # our input for images. Only used in the very end

    class_summary_ = np.unique(labels_, return_counts=True)
    classes_ = class_summary_[0] # an array with the corresponding classes
    class_sizes_ = class_summary_[1] # an array with the number of samples of each class in each entry
    
    # now that we have the number of classes, we want the number of samples in each of the classes to add up to n.
    n_classes_ = len(class_sizes_) 
    class_props_ = class_sizes_ / sum(class_sizes_) # the size of each proportion.
    reduced_class_sizes = class_props_ * n_ # each class should sum to n
    final_reduced_idx = []

    # if we're being given an unrolled, vectorized dataset
    if type(images_) != list: 
        images_ = list(images_)
        
        # we want a list of images to make indexing easier and support outputting a list of images.
        images_ = [images_[i].reshape(28, 28) for i in np.arange(len(images_))] 

    # for each class in the labels, repeat the subsetting procedure.
    for i, class_ in enumerate(classes_): 
        
        # indices where the target is equal to this particular class_. Next, reduce this size.
        class_index_ = np.where(labels_ == class_)[0] 
        red_idx = np.random.choice(class_index_, size=int(reduced_class_sizes[i]), replace=False).tolist()
        final_reduced_idx += red_idx

    final_labels_list = [labels_[i] for i in final_reduced_idx]
    final_images_list = [images_[i] for i in final_reduced_idx]

    if vectorize:
        X_matrix = []
        for image in images:
            image_vector = cv2.resize(image, (32, 32)).flatten() # the size that's required
            X_matrix.append(image_vector)

        final_images_list = np.array(X_matrix, dtype="float") # normalize data
        final_labels_list = np.array(labels)  
    return (final_images_list, final_labels_list)


def multiclass_to_binary(images, labels, target, balance=False):
    '''
    Label transformation.

    inputs:
    ----------
        images (array-like): list of images read into memory

        labels (array-like): a vector containing the labels of the images.

        balanced (bool): Specifies if the negative class (non-target class) 
            should be downsampled to resolve class imbalance. If True, 
            negative response classes will be sampled such that the positive 
            and negative classes are balanced. 

    returns:
    --------
    images_binary (list[array-like]): A subset of images

    labels (list[bool]): depending on the class, or not the class. (Yes or No? Should this be the label? Maybe)
    '''
    y = np.array(labels)
    y_binary = np.where(y == target, True, False)
    if type(images) != list: 
        dims = images.shape ## Is n_images, length, width, channels
        images_binary = list(images)
        if dims == 4:
            images_binary = [images_binary[i].reshape(28, 28, 3) for i in np.arange(len(images_binary))]
        

    if balance==True: # we're betting that the negative class is too large
        pos_idx = np.where(y_binary == True)[0]
        neg_idx = np.where(y_binary == False)[0]

        downsampled_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)

        y_binary = np.hstack((y_binary[pos_idx], y_binary[downsampled_idx]))
        imlist = np.hstack((pos_idx, downsampled_idx)).tolist() # the subset of classes
        images_binary = [images[i] for i in imlist] # assumes images is an array or list
    
    if type(images) != list:
        X_matrix = []
        for image in images_binary:
            image_vector = cv2.resize(image, (32, 32)).flatten() # the size that's required
            X_matrix.append(image_vector)

        images_binary = np.array(images_binary, dtype="float") # normalize data
        y_binary = np.array(y_binary)  
    return images_binary, y_binary


