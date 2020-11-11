
'''
Define how an experiment is conducted for a model, including the validation scheme.
Log all metrics, parameters of the model, etc. using MLflow in this module.

'''

from src import datasets, models, helpers, visualizations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow 
import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AircraftExperiment:
    """
    Run an experiment with choice of Gaussian process model and evaluate it
    using 5-fold cross validation. 
    
    Dataset is the UC Merced land-use land cover dataset. If data_source='local',
    _init_dataset() calls the dataset from data_path in config.yaml. Else, the
    data source can be from AWS S3. 
    """
    
    def __init__(self, model, run_name, experiment_id, projection_title=None, 
                 data_source='local', config_path='config.yaml'):
        """
        
        params:
        -------
            config_path (str): path to config file.
            model (callable): any Model sub-class from models.py to evaluate
        """
        self.config_path = config_path
        self.config = helpers.make_config(config_path)
        self.experiment_id = experiment_id
        self.data_source = data_source
        self.run_name = run_name
        self.model = model
        self.results = None
        self.results_df = None 
        self.projection = None 
        if projection_title == None:
            self.projection_title = run_name
        else:
            self.projection_title = projection_title
            
        # data that will be used in the experiment
        data = datasets.load_image_data(data_path=self.config['data_path'],
                                        classes=self.config['select_classes'],
                                        flatten=False)
        
        self.images, self.labels = data[0], data[1]
 
    def run(self):
        """
        Begin the training and evaluation process for the models.
        """
        with mlflow.start_run(run_name=self.run_name, experiment_id=self.experiment_id):
            # make labels binary based on target
            self.binary_labels = helpers.binarize(y=self.labels, target=self.config['target'])
            
            # score the model with k-fold cv
            self.results = self.evaluate(X=self.images, y=self.binary_labels)
            
            # if we want to save the plots
            if self.config['make_plots']:
                processor = self.model.preprocessor
                embeddings = processor.fit_transform(self.images)
                
                # save figure with figname = run name + run time
                figure_name = f'{self.run_name}_{self.experiment_time}.png'
                
                # save figure path name for mlflow.log_artifact
                self.figure_path = f'figs/{figure_name}'
                
                plt.ioff()
                self.projection = visualizations.plot_feature_vectors(embeddings, 
                                                    self.binary_labels,
                                                    title=self.projection_title,
                                                    method='isomap',
                                                    file_name=figure_name, 
                                                    save_figure=self.config['make_plots'])
                
            # log the stats in mlflow
            self.log_stats(self.results)
            

    def log_stats(self, results):
        """
        log the parameters of the model to MLflow after 5-fold CV.
        performance is measured with mean/std AUC, mean/std accuracy,
        mean/std F1 score, etc.
        
        Also log a low-dimensional projection of the images / embeddings.
        """
        self.results_df = pd.DataFrame(results)
        
        # gaussian process stats
        mlflow.log_metric("Training Time - mean", self.results_df.training_time.mean())
        mlflow.log_metric("Training Time - std", self.results_df.training_time.std())
        
        mlflow.log_metric("Prediction Time - mean", self.results_df.prediction_time.mean())
        mlflow.log_metric("Prediction Time - std", self.results_df.prediction_time.std())
        
        mlflow.log_metric("Accuracy - mean", self.results_df.accuracy.mean())
        mlflow.log_metric("Accuracy - std", self.results_df.accuracy.std())
        
        mlflow.log_metric("Precision - mean", self.results_df.precision.mean())
        mlflow.log_metric("Precision - std", self.results_df.precision.std())
        
        mlflow.log_metric("Recall - mean", self.results_df.recall.mean())
        mlflow.log_metric("Recall - std", self.results_df.recall.std())
        
        mlflow.log_metric("ROC AUC - mean", self.results_df.roc_auc_score.mean())
        mlflow.log_metric("ROC AUC - std", self.results_df.roc_auc_score.std())
        
        mlflow.log_metric("F1 Score - mean", self.results_df.f1_score.mean())
        mlflow.log_metric("F1 Score - std", self.results_df.f1_score.std())
        
        # support vector machine stats
        mlflow.log_metric("SVC Accuracy - mean", self.results_df.svc_accuracy.mean())
        mlflow.log_metric("SVC Accuracy - std", self.results_df.svc_accuracy.std())
        
        mlflow.log_metric("SVC ROC AUC - mean", self.results_df.svc_roc_auc_score.mean())
        mlflow.log_metric("SVC ROC AUC - std", self.results_df.svc_roc_auc_score.std())
        
        mlflow.log_metric("SVC F1 Score - mean", self.results_df.svc_f1_score.mean())
        mlflow.log_metric("SVC F1 Score - std", self.results_df.svc_f1_score.std())
        
        # logistic regression stats
        mlflow.log_metric("LR Accuracy - mean", self.results_df.logistic_accuracy.mean())
        mlflow.log_metric("LR Accuracy - std", self.results_df.logistic_accuracy.std())
        
        mlflow.log_metric("LR ROC AUC - mean", self.results_df.logistic_roc_auc_score.mean())
        mlflow.log_metric("LR ROC AUC - std", self.results_df.logistic_roc_auc_score.std())
        
        mlflow.log_metric("LR F1 Score - mean", self.results_df.logistic_f1_score.mean())
        mlflow.log_metric("LR F1 Score - std", self.results_df.logistic_f1_score.std())
        
        # log the projection figure
        if self.config['make_plots']:
            mlflow.log_artifact(self.figure_path)
        
        # log fitted kernel
        mlflow.log_param('Fitted Kernel', self.model.GP.kernel_)
        
        # log pre-processing pipeline
        mlflow.log_param('Pre-processor', str(self.model.preprocessor.steps))
        
        # log marginal likelihood from most recent model
        loglik = self.model.GP.log_marginal_likelihood_value_
        mlflow.log_param("Log Marginal Likelihood", loglik)
        
        # log configurations
        mlflow.log_params(self.config)
           
        
                    
    def evaluate(self, images, labels):
        """
        5-fold cross validation function to score the model and compare to 
        baseline classifiers (logistic regression, linear/RBF support vector 
        machine).
        
        inputs:
        -------
            X (arr): Training design matrix (images)
            y (arr): Training response vector (labels)
        """
        np.random.seed(self.config['random_seed'])
        
        # baseline models
        svc = SVC(random_state=self.config['random_seed'])
        logistic = LogisticRegression(random_state=self.config['random_seed'])
        
        # define splits for 5-fold CV
        skf = StratifiedKFold(n_splits=self.config['k_folds'], random_state=self.config['random_seed'])
        skf.get_n_splits(X)
        
        results = {'model': [],
                   'training_time': [],
                   'prediction_time': [],
                   'accuracy': [],
                   'precision': [],
                   'recall': [],
                   'roc_auc_score': [],
                   'f1_score': [],
                   'svc_accuracy': [],
                   'svc_roc_auc_score': [],
                   'svc_f1_score': [],
                   'logistic_accuracy': [],
                   'logistic_roc_auc_score': [],
                   'logistic_f1_score': []
                   }
        
        # 5-fold CV loop
        for train_index, test_index in skf.split(images, labels):
            
            # split the data for training and evaluation
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # measure training time
            start_train_time = time.time()
            self.model.train(X_train, y_train)
            training_time = time.time() - start_train_time
                      
                
            # use image transform / feature extraction for LR and SVM
            preprocessor = self.model.preprocessor
            X_train_feat  = preprocessor.fit_transform(X_train)
            X_test_feat = preprocessor.transform(X_test)
            
            svc.fit(X_train_feat, y_train)
            logistic.fit(X_train_feat, y_train)
            
            # prediction time
            start_predict_time = time.time()
            
            # make predictions
            yhat = self.model.predict(X_test)
            
            # time the predictions
            prediction_time = time.time() - start_predict_time
            
            yhat_svc = svc.predict(X_test_feat)
            yhat_lr = logistic.predict(X_test_feat)
            
            # gaussian process scores
            results['model'].append(self.model)
            results['training_time'].append(training_time)
            results['prediction_time'].append(prediction_time)
            results['accuracy'].append(accuracy_score(y_test, yhat))
            results['precision'].append(precision_score(y_test, yhat))
            results['recall'].append(recall_score(y_test, yhat))
            results['roc_auc_score'].append(roc_auc_score(y_test, yhat))
            results['f1_score'].append(f1_score(y_test, yhat))
            
            # support vector machine scores
            results['svc_accuracy'].append(accuracy_score(y_test, yhat_svc))
            results['svc_roc_auc_score'].append(roc_auc_score(y_test, yhat_svc))
            results['svc_f1_score'].append(f1_score(y_test, yhat_svc))
            
            # logistic regression scores
            results['logistic_accuracy'].append(accuracy_score(y_test, yhat_lr))
            results['logistic_roc_auc_score'].append(roc_auc_score(y_test, yhat_lr))
            results['logistic_f1_score'].append(f1_score(y_test, yhat_lr))
        
        # log time of experiment
        self.experiment_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        results['experiment_timestamp'] = [self.experiment_time] * len(results['model'])
        return results