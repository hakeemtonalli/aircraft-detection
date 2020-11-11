'''
Main module for training models with a given kernel, evaluating the models, and saving
the results as MLflow runs. Experiment ID should be "0" until new experiments are 
created using MLflow.
'''

from src import experiments 
from src import models 
from src.helpers import make_config
from sklearn.gaussian_process import kernels as kern
import argparse
import warnings
warnings.filterwarnings("ignore")

params = make_config('config.yaml')


KERNELS = {
    'Linear_Kernel': kern.DotProduct(sigma_0=0) + kern.WhiteKernel(noise_level=0.001),
    'Quadratic_Kernel': kern.DotProduct(sigma_0=0)**2 + kern.WhiteKernel(noise_level=0.001),
    'Cubic_Kernel': kern.DotProduct(sigma_0=0)**3 + kern.WhiteKernel(noise_level=0.001),
    'RBF_Kernel': kern.RBF(),
    'RQ_Kernel': kern.RationalQuadratic()
}


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Provide kernel to test.')
    parser.add_argument('-k', '--kernel', default='RBF_Kernel', required=False)
    parser.add_argument('-n', '--n_restarts', default=3, required=False)
    args = parser.parse_args()
    
    print(f"\nStarting {args.kernel} experiment..\n")
    
    print("Running Vanilla GP..\n")
    experiments.AircraftExperiment(config_path='config.yaml',
                                   experiment_id=params['experiment_id'],
                                   run_name=f'Vanilla GP with {args.kernel}',
                                   projection_title="Original Images",
                                   visualize_multiclass=True,
                                   model=models.VanillaGP(kernel=KERNELS[args.kernel],
                                                     n_restarts_optimizer=args.n_restarts)
                                  ).run()

    print("Running ResNet-18 GP..\n")
    experiments.AircraftExperiment(config_path='config.yaml',
                                   experiment_id=params['experiment_id'],
                                   run_name=f'ResNet-18 GP with {args.kernel}',
                                   projection_title="ResNet-18 Features",
                                   visualize_multiclass=True,
                                   model=models.ResNetGP(kernel=KERNELS[args.kernel],
                                                    n_restarts_optimizer=args.n_restarts)
                                  ).run()

    print("Running SIFT GP..\n")
    experiments.AircraftExperiment(config_path='config.yaml', 
                                   experiment_id=params['experiment_id'],
                                   run_name=f'SIFT GP with {args.kernel}',
                                   projection_title="SIFT Features",
                                   visualize_multiclass=True,
                                   model=models.SIFTGP(kernel=KERNELS[args.kernel], 
                                                   n_restarts_optimizer=args.n_restarts)
                                  ).run()
    
    print("Experiments complete.\n")


    
    
    
