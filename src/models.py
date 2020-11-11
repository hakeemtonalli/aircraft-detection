
'''
    Module for models (algorithms) to be used in the train.py module.
    Experiment class from the experiment.py module is used to evaluate the model
    and save the results of the model to MLflow.
'''

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import kernels as kern
import numpy as np 
from src import datasets, helpers

import warnings
warnings.filterwarnings('ignore')


class VanillaGP:
    """
    Gaussian process with image pre-processing steps, choice of kernel + 
    hyperparameters, but without a feature extraction step. Takes images
    as inputs in the .train() and .predict() methods. 
    
    For more advanced kernels, create a custom kernel in kernels.py, and
    reference that kernel in config.yaml as if it were being called from 
    within this module.
    """
    def __init__(self, kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=3,
                 config_path='config.yaml'):
        """
        params:
        -------
            kernel (callable): Kernel class from sklearn.gaussian_processes module
            optimizer (str or callable): Optimizer for kernel's hyperparameters
            config_path (str): Path to config.yaml file with model/runtime settings
            
        attributes:
        -----------
            config (Dict): Settings for the model + experiments, from config.yaml
            kernel (Kernel): See params
            optimizer (str or callable): See params
            GP (GaussianProcessClassifier): Gaussian process used for classification.
            model (Pipeline): scikit-learn Pipeline with preprocessing steps and model.
        """
        self.config = helpers.make_config(config_path)
        self.kernel = kernel
        self.optimizer = optimizer
        self.GP = GaussianProcessClassifier(kernel=self.kernel, 
                                optimizer=self.optimizer,
                                n_restarts_optimizer=n_restarts_optimizer)
        self.preprocessor = Pipeline([
            ('RGB', datasets.RGBToGrayscale()),
            ('ResizeImages', datasets.ResizeImages(self.config['image_dims'])),
            ('Unroll', datasets.UnrollImages())
        ])
        self.model = Pipeline([
            ('Preprocessor', self.preprocessor),
            ('GaussianProcess', self.GP)
        ])
    
    def train(self, X_train, y_train):
        """
        Train a Gaussian process with a feature extraction pre-processing step. 
        The model takes un-flattened images, processes them, and then feeds the
        resultant vectors into the final classifier.
        inputs:
        -------
            X_train (array): An array of un-flattened images of size (n_images, length, width, 3) 
            to train the Gaussian process. 
            y_train (array): An array of text labels of size (n_images, 1) or 
            (n_images). 
            
        returns:
        --------
            A scikit-learn Pipeline with a trained Gaussian process classifier.
        """
        
        assert len(X_train.shape) == 4, "Need dataset of shape (n_images, length, width, 3)."
        self.model.fit(X_train, y_train)
        
    
    def predict(self, X):
        """
        Make a prediction on a new set of unlabeled data X.
        
        inputs:
        -------
            X (array): Unlabeled dataset of (n_images, length, width, 3) images.

        returns:
        --------
            y_hat (array): Predictions for X.            
        """
        y_hat = self.model.predict(X)
        return y_hat
    

class ResNetGP(VanillaGP):
    '''
    Gaussian process with a ResNet feature extractor pre-processing step and
    choice of kernel with specified hyperparameters. 
    
    image_dims must be (224, 224) for the ResNet feature extractor to work well.
    '''
    def __init__(self, kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=3,
                 config_path='config.yaml'):
        super().__init__(kernel, optimizer, n_restarts_optimizer, config_path)
        self.norm_mean = [0.485, 0.456, 0.406]
        self.norm_std = [0.229, 0.224, 0.225]
        self.preprocessor = Pipeline([
            ('NormalizeData', datasets.NetworkNormalization(mean=self.norm_mean,
                                                            std=self.norm_std)),
            ('ResizeImages', datasets.ResizeImages(self.config['image_dims'])),
            ('ResNet', datasets.ResNetExtractor(cuda=self.config['cuda'], as_array=True))
        ])
        self.model = Pipeline([
            ('Preprocessor', self.preprocessor),
            ('GaussianProcess', self.GP)
        ])

class SIFTGP(VanillaGP):
    """
    Gaussian process with a SIFT feature extractor pre-processing step and
    choice of kernel with specified hyperparameters.
    """
    
    def __init__(self, kernel, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=3,
                 config_path='config.yaml'):
        super().__init__(kernel, optimizer, n_restarts_optimizer, config_path)
        self.preprocessor = Pipeline([
            ('RGB', datasets.RGBToGrayscale()),
            ('ResizeImages', datasets.ResizeImages(self.config['image_dims'])),
            ('SIFT', datasets.SIFTFeatures(n_octave_layers=self.config['n_octave_layers'],
                                        n_clusters=self.config['n_clusters'],
                                        contrast_threshold=self.config['contrast_threshold']))
        ])
        self.model = Pipeline([
            ('Preprocessor', self.preprocessor),
            ('GaussianProcess', self.GP)
        ])
        
