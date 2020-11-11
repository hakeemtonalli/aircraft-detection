'''
    This module is used to create visualizations that summarize the experiment results.
'''

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE
from src import helpers
from mpl_toolkits import mplot3d
import sklearn.gaussian_process.kernels as kern
from sklearn.gaussian_process import GaussianProcessRegressor


def plot_feature_vectors(data, labels, title, file_name, method='isomap', 
                         legend=False, save_figure=False, file_path='figs/'):
    ''' 
    Plot projected vectors in 2-D space with settings helpful for the paper. 
    Returns a matplotlib figure, meaning additional settings can be adjusted. 

    inputs:
    -------
        data (array-like): (n, 2) Design matrix X generated using PCA, Isomap, UMAP, etc.
        labels (array-like[str]): (n, 1) vector y containing the corresponding labels (str) for each sample
        title (str): Title for the plot.
        file_name (str): Path + file name

    '''
    
    if method=='pca':
        pca = PCA(n_components=2, random_state=12)
        data = pca.fit_transform(data)
    elif method=='isomap':
        iso = Isomap(n_components=2, eigen_solver='dense')
        data = iso.fit_transform(data)
    elif method=='kernel_pca':
        kpca = KernelPCA(n_components=2, kernel='rbf', random_state=12)
        data = kpca.fit_transform(data)
    elif method == 'tsne':
        tsne = TSNE(n_components=2)
        data = tsne.fit_transform(data)
    
    df = helpers.viz_to_df(data, labels)
    
    if legend:
        fig = plt.figure(figsize=(12,10))
        sns.scatterplot(x='z1', y='z2', hue='label', data=df)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True, fontsize=24)
    else:
        fig = plt.figure(figsize=(10,10))
        sns.scatterplot(x='z1', y='z2', hue='label', data=df, legend=None)
    
    plt.xlabel("Component 1", size=24)
    plt.ylabel("Component 2", size=24)
    plt.xticks(color=(0,0,0,0))
    plt.yticks(color=(0,0,0,0))
    plt.title(title, size=24)
    plt.tight_layout()
    if save_figure:
        plt.savefig(file_path + file_name)
        
    return fig
        

def plot_class_balance(labels):
    plt.style.use('seaborn')
    """
    inputs:
    -------
        labels: array-like, shape (n_samples)
        Contains the labels of our dataset, to be made into a series and counted.

    returns:
    --------
        Matplotlib bar chart with class balance counts

    """

    class_counts = pd.Series(labels).value_counts()

    plt.figure(figsize=(10,6))
    if len(class_counts) > 6:
        class_counts.plot(kind='barh')
    else:
        class_counts.plot(kind='bar')
        plt.xticks(rotation=45)
    
    plt.title("Number of Images per Class", size=14)
    plt.show()
    print(class_counts)


def image_matrix(image_list, labels, dim=3, random_state=False):
    """
    Plot a dim x dim matrix of images from a list of images. (default 3x3)

    inputs:
    -------
        image_list (list[array-like]) A list of numpy arrays (as opposed to an 
            array of NumPy arrays).

        labels (array-like): A list or array of labels for the images.

    returns:
    -------
        A matrix of images for visualizing the dataset.
    """
    
    total_images = dim * dim
    image_list_size = len(image_list)

    fig, axes = plt.subplots(figsize=(12,8), nrows=dim, ncols=dim)
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])

    for ax, i in zip(axes.flatten(), np.arange(total_images)):
        if random_state != False:
            seed = i + random_state
            np.random.seed(seed)

        rand_image = np.random.randint(low=0, high=image_list_size)
        if len(image_list[rand_image].shape) == 2:
            ax.imshow(image_list[rand_image], cmap=plt.cm.gray)
        else:
            ax.imshow(image_list[rand_image])
        ax.set_title(labels[rand_image])
    plt.tight_layout()
    plt.show()
    
    
def bivariate_gp_samples(kernel, x1, x2, optimizer='fmin_l_bfgs_b', n_samples=1, random_state=12):
    """
    Returns nx2 samples from a Gaussian process, used to visualize a Gaussian
    process in two dimensions.
    """
    
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer,
                                  alpha=1.0, random_state=random_state)


    # we create a xp x yp domain
    Xp, Yp = np.meshgrid(x1, x2)
    
    # flatten the vectors for the GP
    xpp = Xp.flatten()
    ypp = Yp.flatten()
    
    # concatenate the data and transpose for the GP
    data = np.transpose(np.vstack((xpp, ypp)))
    
    # generate samples on the domain
    y_mean, y_std = gp.predict(data, return_std=True)
    y_samples = gp.sample_y(data, 10)
    
    return y_samples

def bivariate_plot(X, Y, Z, title, figpath=None, xlims = [-3,3], ylims = [-3,3]):
    fig = plt.figure(figsize = (8,8))
    """
    Plots the bivariate that was generated from make_bivariate
    
    Inputs:
    -------
        figpath (str): the path to save the figure to. Example: figs/foo.png
    """
    # 3D plot for Gaussian
    fig.add_subplot(1, 1, 1, projection='3d')
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=2, cstride=2, linewidth=0, antialiased=True,
                cmap = 'viridis')
    plt.title(title, size=26)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(axis='x', colors=(0,0,0,0))
    ax.tick_params(axis='y', colors=(0,0,0,0))
    ax.tick_params(axis='z', colors=(0,0,0,0))
    
    if figpath != None:
        plt.savefig(figpath)
        
    plt.show()
    return fig
    
    
def plot_bivariate_gp(kernel, title, optimizer='fmin_l_bfgs_b', granularity=70, figpath=None, n_samples=1, random_state=12):
    """
    Plot samples from a Gaussian process with a given kernel.
    
    inputs:
    -------
        kernel (Kernel): scikit-learn Kernel class to view samples from.
        title (str): The title for the plots.
        figpath (str): Path to output the figure. If None, the figure
        will not be saved. Example: figs/foo.png
        n_samples (int): The number of sample functions to generate.
    """
    plt.ioff()
    xs_ = np.linspace(-4*np.pi, 4*np.pi, granularity)
    Xp, Yp = np.meshgrid(xs_, xs_)
    y_samples = bivariate_gp_samples(kernel=kernel, optimizer=optimizer, x1=xs_, x2=xs_, random_state=random_state)
    
    fig_list = []
    
    for i in np.arange(n_samples):
        z = y_samples[:,i].reshape(granularity,granularity)
        fig = bivariate_plot(X=Xp, Y=Yp, Z=z, title=title, figpath=figpath,
                       xlims=[-4*np.pi,4*np.pi], ylims=[-4*np.pi, 4*np.pi])
        if n_samples == 1:
            return fig
        else:
            fig_list.append(fig)
            
    return fig_list
            



def plot_gram_matrix(kernel, optimizer='fmin_l_bfgs_b', title="Covariance Matrix", figpath=None):
    '''
    Plot the gram matrix for a given kernel.
    
    Inputs:
    -------
        kernel (Kernel): scikit-learn Kernel class from the gaussian_processes
        optimizer (str): Optimizer to use
        
    '''
    x = np.linspace(0, 1, 100)
    plt.figure(figsize = (8,6))
    ax = sns.heatmap(kernel(x.reshape(-1,1), x.reshape(-1,1)),cmap = 'GnBu', vmin=0, vmax=1)
    ax.set_title(title, size = 24)
    ax.axis("off")
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=22)
    
    if figpath != None:
        plt.savefig(figpath)

    plt.show()

def plot_similarity_decay(kernel, optimizer='fmin_l_bfgs_b', title='Similarity Decay', figpath=None):
    x = np.linspace(0, 1, 100)
    plt.figure(figsize = (6,6))
    plt.plot(x, kernel(x.reshape(-1,1), x.reshape(-1,1))[0], lw = 3)
    plt.title(title, size=24)
    plt.xlabel("x", size=24)
    plt.ylabel("$k(x,x')$", size=24)
    plt.xticks(color=(0,0,0,0))
    plt.yticks(color=(0,0,0,0))
    
    if figpath != None:
        plt.savefig(figpath)
        
    plt.show()

def plot_univariate_gp(kernel, title, optimizer='fmin_l_bfgs_b', figpath=None, n_samples=5):
    x = np.linspace(0, 1, 100)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0, optimizer=optimizer)
    y_mean, y_cov = gp.predict(x[:,np.newaxis], return_std=True)
    y_samples = gp.sample_y(x[:, np.newaxis], n_samples)
    
    plt.figure(figsize = (6,6))
    plt.fill_between(x, y_mean - 1.96*y_cov, y_mean + 1.96*y_cov, alpha=0.15, color='k')
    plt.plot(x, y_mean, color='navy', lw=3)
    plt.plot(x, y_samples, color='k', alpha=0.35, lw=3)
    plt.ylim(-3, 3)
    plt.xticks(color=(0,0,0,0))
    plt.yticks(color=(0,0,0,0))
    #plt.legend(loc='upper right', facecolor='white', frameon=True, framealpha=0.5)
    plt.title(title, size=24)
    
    if figpath != None:
        plt.savefig(figpath)
        
    plt.show()
    
    
def plot_sift_descriptors():
    """
    Plot the SIFT descriptors on a given image. To be used in
    the SIFT feature extractor to see a couple of examples of the 
    make_sift() and self.model methods to debug.
    """
    pass