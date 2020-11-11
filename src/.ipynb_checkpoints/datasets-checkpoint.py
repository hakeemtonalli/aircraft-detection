
from imutils import paths
import random 
import os
import torch 
from torch import nn
import torchvision.models as torchmodels
import numpy as np 
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from skimage import feature
from pathlib import Path
import tempfile
import boto3
import warnings
warnings.filterwarnings('ignore')

class SIFTFeatures(BaseEstimator, TransformerMixin):
    '''

    SIFT visual bag of words (SIFT-VBOW) feature extractor packaged as a scikit-learn
    Transformer for turning images into histograms of cluster labels, where the length
    of the histogram is the number of clusters K chosen. 
     
    Uses OpenCV implementation of SIFT with K-Means for creating vector representations 
    of the dataset of images.
    
    Requires a (n_images, length, width) shape dataset (unflattened). 

    params:
    -------
        nOctaveLayers (int): Number of layers per octave (default at 60)
        
        contrastThreshold (float): Parameter that controls the number of features detected.
        Keep at a low threshold (0.01 for example) to detect a lot (although weak) of 
        features. May return uninteresting features (low-contrast) if the threshold is 
        too low
        
    attributes:
    -----------
        n_clusters (int): The size of the histogram output. Also is thought of as the number 
        of feature "groups" that are in the data set of images. Can be found using hyperparameter
        search / cross-validation. 
        
        sample_images (list): List of original images before being stored using
        a SIFT descriptor.
        
        sift_images (list): List of images being stored after using SIFT detector.
        Helpful for debugging.
 
        
    '''
    def __init__(self, n_octave_layers=60, contrast_threshold=0.01, n_clusters=500):
        self.n_octaveLayers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.n_clusters = n_clusters
        self.sample_images = []
        self.sift_images = []
        return 
    
    def make_sift(self, X, y=None):
        """
        Generate SIFT descriptors from NumPy array. Each 

        input
        -----
        X: single-channel numpy array with pixel values from 0 to 1. 
        size: n_images, length, width (not flattened)

        parameters
        ----------
        nOctaveLayers: Number of layers per octave (default at 60)
        contrastThreshold:

        """

        img_descs = []
        im_indices = np.random.randint(low=1, high=len(X), size=4)
        
        for image in X:
            
            # SIFT complains if image is not uint8
            formatted_image = (image*255).astype('uint8')
            
            # define SIFT feature extractor
            sift = cv2.xfeatures2d.SIFT_create(n_octave_layers=self.n_octave_layers, contrastThreshold=self.contrast_threshold)
            
            # extract keypoints and descriptors
            kp, desc = sift.detectAndCompute(formatted_image, None)
            
            if i in im_indices:
                self.sample_images.append(image)
                self.sift_images.append(cv2.drawKeypoints(formatted_image, kp, None, color=(255, 0, 0)))
            
            # if no keypoints are detected, make zeros
            if len(kp) < 1:
                desc = np.zeros((1, sift.descriptorSize()), np.float32)
                
            # the descriptors should be 128 in dimensionality
            assert desc.shape[1] == 128
            img_descs.append(desc)
            
        X_descriptors = np.concatenate(img_descs)
        
        # store descriptors for debugging
        self.descriptors = img_descs
        return X_descriptors

    def fit(self, X, y=None):
        '''
        Apply K-means clustering to the bag of SIFT words to find clusters of features
        that appear in the dataset.
        
        Creates a K-Means model that is used to quantize the descriptors detected in 
        an image.
        
        input:
        ------
            X (array): Shape (n_images, length, width). 
        
        '''
        self.model_ = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=12)
        X_descriptors = self.make_sift(X)
        self.model_.fit(X_descriptors)
        return self
    
    def transform(self, X, y=None):
        '''
        Use the trained K-means model to classify each descriptor (assign to a cluster) and
        create a histogram for each image. 
        
        input:
        ------
            X (array): Dataset of images of shape (n_images, length, width).
            
        returns:
        --------
            X_hist (array): (n_images, k) matrix of SIFT-VBOW features representing the original dataset.
        '''
        cluster_model = self.model_ 
        
        descriptors = self.make_sift(X) 

        img_clustered = [cluster_model.predict(descriptor) for descriptor 
                         in self.descriptors] 
        
        sift_features = [np.bincount(clustered_words, minlength=self.n_clusters) 
                         for clustered_words in img_clustered]
        
        sift_features = np.array(sift_features)
        return sift_features



class ResNetExtractor(BaseEstimator, TransformerMixin):
    ''' 
    
    ResNet-18 based Feature extractor packaged as a sklearn Transformer object
    to convert images into ResNet embeddings. Inputs can be ndarrays or Tensors, 
    and the .transform() method can output an ndarray or Tensor.
    
    However, if using this object in a scikit-learn pipeline, be sure to set 
    as_array=True to output an array for the next link in the pipeline.
    
    Outputs (n, 512) compressed feature vector.
    Requires 224 x 224 input
    
    
    params:
    -------
        cuda (bool): Use GPU (cuda=True) or CPU (cuda=False).
        as_array (bool): Return a numpy ndarray instead of a pytorch Tensor. 
        Recommended for use with scikit-learn Pipelines.
    '''

    def __init__(self, cuda=False, as_array=False):
        self.device = "cuda" if cuda else "cpu"
        self.as_array_ = as_array
        self.feature_extractor = self._init_feature_extractor().double()

    def fit(self, images, labels=None):
        return self 

    def transform(self, images, labels=None):
        '''
        
        Take a matrix of images (ndarray or Tensor), convert to Tensor if necessary,
        and pass it through the ResNet-18 feature extractor. Returns embeddings as 
        numpy ndarrays or pytorch Tensors. 
        
        inputs:
        -------
            images (array-like): numpy array or pytorch Tensor with the original
            data. Needs to be n_images x 3 x 224 x 224, or n_images x 224 x 224 x 3
            for Tensor input. For best results (theoretically) make sure that the images
            are normalized with mu = [0.485, 0.456, 0.406], sigma = [0.229, 0.224, 0.225].
        '''
        # when input is ndarray
        if type(images) is np.ndarray:
            images = np.swapaxes(images, 1, 3)
            images = torch.from_numpy(images)
            images = images.double().to(self.device)
            
        # extract features. Takes a long time, unless on GPU and/or parallelized. 
        embeddings = self.feature_extractor(images)

        # turn into array if as_array
        if self.as_array_:
            # convert to numpy
            embeddings = embeddings.numpy()
            
            # knock off extra dimensions from Tensor
            embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1])
        
        return embeddings

    def _init_feature_extractor(self):
        ''' 
        Initialize ResNet-18 feature extractor. Pop off the last layer of the ResNet
        and return it. 
        '''
        # get pre-trained torchvision model
        model = torchmodels.resnet18(pretrained=True)
        model = model.to(self.device)

        # get feature extractor and freeze layers
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        return feature_extractor


    
class HOGTransform(BaseEstimator, TransformerMixin):
    '''
    Histogram of oriented gradients (HOG) implementated as a scikit-learn Transformer.
    
    Used to compare SIFT baseline.  
    
    Uses scikit-image feature.hog() in the .transform() method.
    '''
    def __init__(self, pixels_per_cell=(3,3), cells_per_block=(2,2), transform_sqrt=True, block_norm='L1'):
        self.pixels_per_cell = pixels_per_cell 
        self.cells_per_block = cells_per_block
        self.transform_sqrt = transform_sqrt 
        self.block_norm = block_norm 
        return 

    def fit(self, X, y=None):
        return self 
    

    def transform(self, X, y=None):
        '''
        input:
        ------
            X (array): array of single-channel images (grayscale). 
        
        returns:
        --------
            X_hog (arrray): array of HOG-transformed images. 
        
        '''
        
        

        assert len(X.shape) >= 3, "Input must be three channels."

        data = []
        labels = []
        for i in range(len(X)):
            image = X[i,:]
            label = y[i]
            
            H = feature.hog(image, orientations=9, pixels_per_cell=(3,3),
                        cells_per_block=(2,2), transform_sqrt=True, block_norm='L1')
            data.append(H)
            labels.append(label)

        X_hog = np.array(data)
        nans_ = np.isnan(X_hog)
        X_hog[nans_] = 0
        return X_hog

class ResizeImages(BaseEstimator, TransformerMixin):
    def __init__(self, new_dim):
        '''
        Resize image to new_dim x new_dim. Useful to use before SIFTFeatures() in the pipeline
        to test the effect of image resolution on the accuracy of our predictions.
        
        params:
        -------
            new_dim (tuple): The new (length, width) dimensions of the image. 
        

        '''
        self.new_dim = new_dim
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        '''
        inputs:
        -------
            X (array): Input design matrix with the raw images. Must be shape of
            (n_images, length, width), or (n_images, length, width, channels). 
            Otherwise it will scream at you.

        returns:
        --------
            resized_imagesd (array): Design matrix with resized images 
            (new_dim[0] x new_dim[1]) 
        '''
        
        resized_images = []
        
        # either len 3 (1 channel) or len 4 (3 channel)
        n_dimensions = len(X.shape) 

        assert n_dimensions >= 2, "Dimension Mismatch."

        # for each image in the data set, resize and build new array
        for img in X:
            new_img = cv2.resize(img, (self.new_dim[0], self.new_dim[1]), 
                                 interpolation=cv2.INTER_AREA) 
            resized_images.append(new_img)
            
        # stack all vectors into an array
        resized_images = np.array(resized_images) 
        return resized_images

class NetworkNormalization(BaseEstimator, TransformerMixin):
    '''
    Normalize input data to follow mean and standard deviation 
    conventions for neural network implementations (VGG and ResNet).
    
    Default for VGG is mu = 0.485 and sigma = 0.229. The default for
    ResNet-18 is mu = [0.485, 0.456, 0.406] and sigma=[0.229, 0.224, 0.225].
    
    params:
    -------
        mean (float or array-like): Centering factor (mean)
        std (float or array-like): Compression factor (standard deviation)
    '''
    
    def __init__(self, mean=0.485, std=0.229):
        self.mean = mean
        self.std = std
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = (X - self.mean) / self.std ## divide by standard deviation
        return X_


class RGBToGrayscale(BaseEstimator, TransformerMixin):
    '''
    Convert a 3-color numpy array to a single-channel grayscale numpy array. 
    Implemented as a scikit-learn Transformer, helpful for Pipelines.
    '''
    def __init__(self):
        return
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        '''
        Each image in X must be a 3-channel image, otherwise an error will
        be raised.
        
        Inputs:
        -------
            X (array): 3-channel array to convert to grayscale.
            
        Returns:
        --------
            X_ (array): 1-channel grayscale array.
        '''
        
        assert len(X.shape) >= 3, "Function transforms RGB images only."
        
        if len(X.shape) < 4:
            r = X[:,:,0]
            g = X[:,:,1]
            b = X[:,:,2]
        else:
            r = X[:,:,:,0]
            g = X[:,:,:,1]
            b = X[:,:,:,2]
        grayscale_images = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return grayscale_images

class UnrollImages(BaseEstimator, TransformerMixin):
    '''
    Scikit-learn Transformer for flattening a (n_images, length, width) dataset
    of images for the classification of raw images. Used in a pipeline.
    '''
    def __init__(self):
        return
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        '''
        Each image must be a 3-channel image.
        '''
        n_images = len(X)
        X_ = X.reshape((n_images, -1))
        return X_

class BalanceClasses(BaseEstimator, TransformerMixin):
    '''
    Take a binary class dataset and make the majority class the size of the 
    minority class. Used for training, to test the effect of balancing 
    classes (specifically via down-sampling in the first verrsion).  
    
    X and y are required, so that the discarded indices from y are used to
    also discard the appropriate samples in X, or so that synthetic observations
    are used in the right places.
    
    When method='downsample', the .transform() method just downsamples the majority 
    class so that len(majority) = len(minority). When method='upsample', SMOTE will
    be used to create synthetic entries. method='upsample' is not implemented yet.
    
    To be used at the head of the Pipeline, as it only takes 3-channel images.
    
    '''
    def __init__(self, method='downsample'):
        self.method = method

    
    def fit(self, X, y):
        return self 
    

    def transform(self, X, y):
        '''
        Each image must be a 3-channel image.
        
        inputs:
        -------
            X (array): Input matrix of shape (n_images, length, width, 3)
        '''
        if self.method == 'downsample':
            X_, y_ = self.downsample(X, y)
        else:
            raise NotImplementedError
        
        return X_, y_ 
    
    def downsample(self, X, y):
        '''
        Helper function to downsample the majority class to match the number of observations
        in the minority class.
        '''
        # resample dataset

        class_counts = np.unique(y, return_counts=True)
        classes = class_counts[0]
        class_sizes = class_counts[1]

        minority_class_index = np.argmin(class_sizes)
        minority_class = classes[minority_class_index]
        minority_size = class_sizes[minority_class_index]

        # the index of each row vector that is of the majority class
        majority_index = (y != minority_class).reshape(-1) 
        
        # the index of each row vector that is in the minority class
        minority_index = (y == minority_class).reshape(-1) 

        X_majority = X[majority_index] 
        y_majority = y[majority_index]
        X_minority = X[minority_index]
        y_minority = y[minority_index]

        X_majority_resampled, y_majority_resampled = resample(X_majority, y_majority, 
                                                              replace=False, n_samples=minority_size, 
                                                              random_state=12)

        X_ = np.vstack((X_minority, X_majority_resampled))
        y_ = np.concatenate((y_minority, y_majority_resampled))
        assert len(X_) == len(y_)
        p = np.random.permutation(len(X_))

        X_ = X_[p]
        y_ = y_[p]
        return (X_, y_)

class DataLoaderToArray(object):
    ''' 
    To be used after dataloader, to convert dataloader into NumPy array.
    
    Preferable for small (n < 80k) experiment data. Not recommended for large 
    datasets. Essential for our experiments, since we use scikit-learn and want 
    a fair comparison (raw data, standardized, SIFT, ResNet), meaning the same 
    dataset used in a PyTorch model (which utilizes the Dataloader). 
    '''
    def __init__(self, combine=False):
        self._combine = combine
        
    def convert(self, dataloader):
        '''
        Return images and labels separately if combine=False, else return only
        the array.
        
        inputs:
        -------
            dataloader (torch.DataLoader): With the data set we want to use
            
        returns:
        --------
            (images, labels): A tuple with the images (X matrix) and labels (y vector).
        '''
        images = []
        labels = []
        for ims, labs in dataloader:
            images.append(ims)
            labels.append(labs)
        images = torch.cat(images).numpy()
        labels = torch.cat(labels).numpy()

        class_mapping = dataloader.dataset.class_to_idx
        class_mapping = dict(zip(class_mapping.values(), class_mapping.keys())) ## reverse the order

        # transform images and labels
        images = np.moveaxis(images, 1, 3)
        labels = np.array([class_mapping[x] for x in labels])
        
        if self._combine:
            raise NotImplementedError("Combining data not supported yet.")
        return images, labels
        
def load_image_data(classes='all', data_path='../data/', grayscale=False, 
                    as_lists=False, flatten=False, dim=(256,256), random_state=12):
    
    '''
    Get local image data. Narrow down to certain classes, resize (using dim keyword arg)
    and return as a matrix and label numpy array. Optional: turn images to grayscale
    and flatten them as long vectors.
    
    inputs:
    -------
        data_path (str): Path for dataset. Default to '..'/data/'
        classes (str or list[str]): The classes to be included in the dataset. Default: 'all'
        grayscale (bool): Returns grayscale images. Default: False
        as_lists (bool): Return images, labels as two lists of arrays and strings 
            instead of two ndarrays.
        flatten (bool): Turn (length, width) image to a (1 x length x width) vector.
            Recommended True only when . flatten=False is useful if 
            only interested in visualizing the data or the application calls for
            a (n_images, length, width) input. 

    returns:
    -------
        X (array-like): Design matrix
        y (array-like): label vector (a list or vector of string labels)
    '''
    
    # the classes in our dataset are the names of the folders
    all_classes = os.listdir(data_path)

    # image paths
    images = paths.list_images(data_path) 

    # sort image paths, and then shuffle them.
    imagePaths = sorted(list(images))
    random.seed(12)
    random.shuffle(imagePaths)

    images = []
    labels = []
    X_matrix = []
    
    if classes == 'all':
        classes = all_classes 

    # iterate over image paths and load into lists
    for imagePath in imagePaths:
        
        # the name of the folder containing the image
        label = imagePath.split(os.path.sep)[-2] 
        
        # narrow down images to selected labels
        if label in classes: 
            
            # turn to grayscale
            if grayscale==True:
                image = cv2.imread(imagePath, 0)
            else:
                image = cv2.imread(imagePath)
            
            # need each image to have length, width, channels
            assert len(image.shape) == 3, "Dimension mismatch: {}".format(str(image.shape))

            if flatten:
                unrolled_image = cv2.resize(image, dim).flatten()
            else:
                unrolled_image = cv2.resize(image, dim)
                
            X_matrix.append(unrolled_image)
            
            # we want to return a list of images: each element is a 3-channel image (array)
            images.append(image) 
            labels.append(label)

        # skip images that aren't in our classes             
        else:
            continue

    # convert lists of images, labels to arrays.
    if as_lists == False:
        images = np.array(X_matrix) / 255.0 # normalize data
        labels = np.array(labels)

    return images, labels


def get_s3_dataset(bucket_name='mri-dataset', classes='all', as_lists=False, destination=None):
    '''
    This function downloads the MRI dataset from its s3 bucket on AWS. We can 
    include desired classes or all available classes.
    
    If keep=True, the dataset is saved to a folder designated by "destination" 
    keyword arg.
    
    It uses the client method, but will use the resource method next to speed 
    things up. This function grabs one image at a time and builds up a list 
    in memory. 
    
    
    inputs:
    -------
        bucket_name (str): Name of the S3 bucket that contains the data set
        classes (str): Which classes to include. Defaults to 'all', meaning no 
            particular classes will be omitted.
        as_lists (bool): Returns images and labels in lists. Default to False,
            so that the function returns an array of images.
        destination (str): the path to save the dataset from S3 (if we want to
            save the images locally after loading them in memory). Default to
            None so that the images are only kept in memory, and not in storage.
    
    '''
    
    local_path = os.path.dirname(os.path.realpath(__file__)) # To save the images in our local directory.
    #### IDENTIFY DATA IN THE BUCKET ####
    
    # initiate s3 resource
    client = boto3.client('s3')
    resource = boto3.resource('s3')
    my_bucket = resource.Bucket(bucket_name)

    # get the labels and filenames from the s3 bucket
    labels = []
    filenames = []
    for s3_object in my_bucket.objects.all():
        # the directory is bucket_name/label/filename
        label, filename = os.path.split(s3_object.key)
        if (label in classes) or (classes=='all'): # only want it if its in a designated class
            if filename != '': # want to ignore the folder name. could add to the first if statement
                labels.append(label) # update list with labels
                filenames.append(filename) # update list with file names

    # -- download data -- ##
    
    unique_labels = list(set(labels)) # to create the directories
    images = [] # for our images list

    # Load all files from s3 using the client method. Modify later for speed
    for label, filename in zip(labels, filenames): # iteratively grab one object at a time
        obj_name = label + '/' + filename ## for downloading
        obj = my_bucket.Object(obj_name) ## for storing in memory

        if destination != None: # if we want to download the file
            for unique_label in unique_labels:
                Path(local_path + '/data/' + unique_label).mkdir(parents=True, exist_ok=True) # to store data by label
            client.download_file(bucket_name, obj_name, 'data/' + obj_name) # slow client method to download the data
        
        # if we don't want to keep, skip previous few lines:
        tmp = tempfile.NamedTemporaryFile()
        with open(tmp.name, 'wb') as f: # storing image in memory, one at a time
            obj.download_fileobj(f)
            img = cv2.imread(tmp.name) # store as image. comes from old script.
            images.append(img)
    return images, labels
    
def get_mri_dataset(classes='all', as_lists=False, destination=None, keep=False):
    return get_s3_dataset(bucket_name="mri-dataset", classes=classes, as_lists=as_lists, destination=destination)

def get_mnist_dataset(classes='all', as_lists=False, destination=None, keep=False):
    return get_s3_dataset(bucket_name="mnist-dataset", classes=classes, as_lists=as_lists, destination=destination)

def get_landuse_dataset(classes='all', as_lists=False, destination=None, keep=False):
    return get_s3_dataset(bucket_name="landuse-dataset", classes=classes, as_lists=as_lists, destination=destination)

def get_cifar_dataset(classes='all', as_lists=False, destination=None, keep=False):
    return get_s3_dataset(bucket_name="cifar-dataset", classes=classes, as_lists=as_lists, destination=destination)