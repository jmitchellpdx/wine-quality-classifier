# src/pipeline_refactored.py
"""
Refactored Wine quality classification modeling pipeline.
Each model has its own explicit method for clarity and understanding.
"""

import pandas as pd
import numpy as np

from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve,auc

import matplotlib.pyplot as plt
import seaborn as sns



# Get convenience functions from utils
from utils import (
    load_wine_data, create_binary_target, prepare_features_and_target,
    split_and_scale_data, print_data_summary, encode_wine_type,print_section_header
)

class WineModelPipeline:
    """
    Wine quality classification pipeline with explicit model methods.
    Each model type has its own method for transparency and understanding.
    """
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config (WineConfig): Configuration object with all parameters
        """
        self.config = config
        self.config.validate_config()
        
        # Storage for data
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        
        # Storage for trained models and results
        # Structure: self.models[model_name] = {model, test_accuracy, f1_score, search_insights, predictions, probabilities, class_report}
        # Search insights is a small dict of outputs from the search
        # Predictions are the binary predictions from the model 
        # probabilities are the class probabilities 
        self.models = {}
        
    
    def load_and_prepare_data(self, verbose = True):
        """
        Load wine data and prepare it for modeling.
        Must be called before any model training methods.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test) - prepared data splits
        """

        print_section_header("LOADING AND PREPARING DATA")

        # Load dataset
        self.df = load_wine_data(self.config.processed_data_path)
        
        # Encode categorical features (wine_type -> binary)
        self.df = encode_wine_type(self.df)
        
        # Create binary target
        df_with_target = create_binary_target(self.df, self.config)
        
        # Prepare features and target
        X, y = prepare_features_and_target(df_with_target, self.config)
        
        # Split and scale data
        (self.X_train, self.X_test, self.y_train, self.y_test, 
         self.scaler, self.label_encoder) = split_and_scale_data(X, y, self.config)
        
        # Store feature names for later use
        self.feature_names = list(self.X_train.columns)
        
        # Print summary
        if verbose:
            print_data_summary(self.X_train, self.X_test, self.y_train, self.y_test, self.label_encoder)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def find_best_logistic_model(self):
        """
        Find best logistic regression model using grid search.
        Stores the model and key insights from the search process.

        Uses parameters in WineConfig:
            Base parameters
            * penalty: loss function
            * max_iter': 1000 
            * solver: what solver to use (e.g saga)
            * class_weight: set to 'balanced' to compensate for 'regular' vs 'premium' class imbalance.

            For the grid search:
            * C: Controls regularization (large C -> weaker regularization, coefficients can grow larger)

        Returns:
            dict: Model results including trained model and performance metrics
        """
        print_section_header("FINDING BEST LOGISTIC REGRESSION MODEL")

        # Get base parameters (fixed) and hyperparameters (for grid search) from config

        base_params = self.config.get_base_params('logistic')
        param_grid = self.config.get_grid_params('logistic')
     
        # Create base model, passing in the dictionary of base parameters
        base_model = LogisticRegression(**base_params)

        # GridSearchCV will test all combinations in param_grid using cross-validation
        # the parameters for this model will be used, others ignored.
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=-1,
            verbose=self.config.grid_search_verbosity
        )
        
        start_time = datetime.now()
        grid_search.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
    
    
        # Save insights from grid search
        search_insights = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'total_fits': len(grid_search.cv_results_['params']),
            'param_combinations_tested': param_grid,
            'training_time_seconds': training_time
        }
 
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_

        # Find predicted classes (0/1) 
        y_pred = best_model.predict(self.X_test)
        
        # Extract probabilities for ROC analysis ([:, 1] gets premium class probabilities)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]  
        
        # accuracy, f1
        test_accuracy = best_model.score(self.X_test, self.y_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # precision and recall etc can be extracted from classification_report,
        # so I'll save that too.
        report = classification_report(self.y_test, y_pred, output_dict=True)
    
        
        # Store everything
        self.models['logistic'] = {
            'model': best_model,
            'search_insights': search_insights,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'class_report': report
        }


        # Display summary
        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best parameters: {search_insights['best_params']}")
        print(f"Best CV score: {search_insights['best_cv_score']:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")

        if self.config.pipeline_verbose:
            print("\n Additional Search Details")
            print(f"  Total fits performed: {len(grid_search.cv_results_['params'])}")
            print(f"Parameter grid: {param_grid}")
        
        return self.models['logistic']
    
    def find_best_rf_model(self):
        """
        Find best Random Forest model using grid search.

        Uses parameters in WineConfig:
        
            Base parameters:
            * n_jobs: set to -1 to use parallel CPUs (if available)
            * class_weight: set to 'balanced' to compensate for 'regular' vs 'premium' class imbalance.
            * random_state: initial value for randomization

            For the grid search:
            * n_estimators: Number of trees
            * max_depth: Maximum tree depth (None = unlimited)
            * max_features: Number of features to consider at each split (either sqrt or log_2 )
        
        Returns:
            dict: Model results including trained model and performance metrics
        """

        print_section_header("FINDING BEST RANDOM FOREST")

        base_params = self.config.get_base_params('rf')
        param_grid = self.config.get_grid_params('rf')
     
        # Create base model, passing in the dictionary of base parameters
        base_model = RandomForestClassifier(**base_params)

        # Create and run grid search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=-1,
            verbose=self.config.grid_search_verbosity
        )
        
        start_time = datetime.now()
        grid_search.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
    
    
        # Save insights from grid search
        search_insights = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'total_fits': len(grid_search.cv_results_['params']),
            'param_combinations_tested': param_grid,
            'training_time_seconds': training_time
        }
 
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        # Find predicted classes (0/1) as well as class probabilities
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]
        
        # accuracy, f1
        test_accuracy = best_model.score(self.X_test, self.y_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # precision and recall etc can be extracted from classification_report,
        # so I'll save that too.
        report = classification_report(self.y_test, y_pred, output_dict=True)
    
    
        
        # Store everything
        self.models['rf'] = {
            'model': best_model,
            'search_insights': search_insights,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'class_report': report
        }


        # Display summary
        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best parameters: {search_insights['best_params']}")
        print(f"Best CV score: {search_insights['best_cv_score']:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")

        if self.config.pipeline_verbose:
            print("\n Additional Search Details")
            print(f"  Total fits performed: {len(grid_search.cv_results_['params'])}")
            print(f"  Parameter grid: {param_grid}")
        
        return self.models['rf']
    
    def find_best_svm_model(self):
        """
        Find best Support Vector Machine (SVM) model using grid search.
        
        Uses parameters in WineConfig:
        
            Base parameters:
            * kernel: kernel to use (such as rbf)
            * class_weight: set to 'balanced' to compensate for 'regular' vs 'premium' class imbalance.
            * random_state: initial value for randomization

            For the grid search:
            * C: Regularization (lower C gives smoother boundaries, large C complex boundaries)
            * gamma: kernel width (higher gives more complex boundaries)
        
        Returns:
            dict: Model results including trained model and performance metrics
        """
        
        print_section_header("FINDING BEST SVM")
     
        base_params = self.config.get_base_params('svm')
        param_grid = self.config.get_grid_params('svm')
     
        # Create base model, passing in the dictionary of base parameters
        base_model = SVC(**base_params)

        # Create and run grid search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=-1,
            verbose=self.config.grid_search_verbosity
        )
        
        start_time = datetime.now()
        grid_search.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
    
    
        # Save insights from grid search
        search_insights = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'total_fits': len(grid_search.cv_results_['params']),
            'param_combinations_tested': param_grid,
            'training_time_seconds': training_time
        }
 
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
       # Find predicted classes (0/1) as well as class probabilities
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]
        
        # accuracy, f1
        test_accuracy = best_model.score(self.X_test, self.y_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # precision/recall measures are in the classification report.
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        # Store everything
        self.models['svm'] = {
            'model': best_model,
            'search_insights': search_insights,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'class_report': report
        }

        # Display summary
        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best parameters: {search_insights['best_params']}")
        print(f"Best CV score: {search_insights['best_cv_score']:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")

        if self.config.pipeline_verbose:
            print("\n Additional Search Details")
            print(f"  Total fits performed: {len(grid_search.cv_results_['params'])}")
            print(f"  Parameter grid: {param_grid}")
        
        return self.models['svm']
    
    
    def find_best_mlp_model(self):
        """
        Find best Multi-Layer Perceptron (Neural Network) model using grid search.

        Uses parameters in WineConfig:
        
            Base parameters:
            * max_iter: Maximum iterations 
            * early_stopping: whether to stop early (to prevent overfitting)
            * n_iter_no_change: how long to keep going without improvement on validation data
            * validation_fraction: fraction of data to use for validation.

            For the grid search:
            * hidden_layer_sizes: structure of internals of MLP (e.g (100,) = 1 layer with 100 neurons)
            * alpha: L2 regularization strength (higher = more regularization)
            * learning_rate_init: Initial learning rate for optimization

        
        Returns:
            dict: Model results including trained model and performance metrics
        """
        
        print_section_header("FINDING BEST MLP MODEL")
        

        base_params = self.config.get_base_params('mlp')
        param_grid = self.config.get_grid_params('mlp')
     
        # Create base model, passing in the dictionary of base parameters
        base_model = MLPClassifier(**base_params)

        # Create and run grid search
        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=self.config.cv_folds,
            scoring=self.config.scoring_metric,
            n_jobs=-1,
            verbose=self.config.grid_search_verbosity
        )
        
        start_time = datetime.now()
        grid_search.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
    
    
        # Save insights from grid search
        search_insights = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'total_fits': len(grid_search.cv_results_['params']),
            'param_combinations_tested': param_grid,
            'training_time_seconds': training_time
        }
 
        # Get the best model from the grid search
        best_model = grid_search.best_estimator_
        
        # Find predicted classes (0/1) as well as class probabilities
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1]
        
        # accuracy, f1
        test_accuracy = best_model.score(self.X_test, self.y_test)
        f1 = f1_score(self.y_test, y_pred, average='weighted')

        # precision/recall measures are in the classification report.
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # Store everything
        self.models['mlp'] = {
            'model': best_model,
            'search_insights': search_insights,
            'test_accuracy': test_accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'class_report': report
        }

        # Display summary
        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best parameters: {search_insights['best_params']}")
        print(f"Best CV score: {search_insights['best_cv_score']:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")

        if self.config.pipeline_verbose:
            print("\n Additional Search Details")
            print(f"  Total fits performed: {len(grid_search.cv_results_['params'])}")
            print(f"  Parameter grid: {param_grid}")
        
        return self.models['mlp']


    
    def train_selected_models(self):
        """
        Convenience method to train a selection of models

        Uses parameters in WineConfig:
            * models_to_run: one or more models from 'rf', 'svm', 'logistic', 'mlp'

        """
        print_section_header("TRAINING MODELS")
        
        # Determine which models to run from config file.
        models_to_run = self.config.get_models_to_run()
        
        print(f"From Configuration file loaded: {models_to_run}")
        
        # Iterate over the list.
        for i, model_name in enumerate(models_to_run, 1):
            
            try:
                if model_name == 'logistic':
                    self.find_best_logistic_model()
                elif model_name == 'rf':
                    self.find_best_rf_model()
                elif model_name == 'svm':
                    self.find_best_svm_model()
                elif model_name == 'mlp':
                    self.find_best_mlp_model()
                else:
                    print(f"Unknown model: {model_name}")
                    
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
        
    
    def print_model_training_results(self):
        """
        Print ranked summary of trained models.
        """
        print_section_header("MODEL TRAINING SUMMARY")
        
        # Sort by f1 score as the best one stat ranking
        model_items = list(self.models.items())
        model_items.sort(key=lambda x: x[1].get('f1_score', 0), reverse=True)
        
        # Summarize the models in a simple table
        print(f"{'Model':<12} {'F1 Score':<12} {'Test Accuracy':<12} {'CV Score':<12}  {'Time(s)':<12}")
        print("-" * 65)
        
        for model_name, results in model_items:
            test_accuracy = results.get('test_accuracy', 0)
            cv_score = results['search_insights'].get('best_cv_score', 0)
            f1 = results.get('f1_score', 0)
            time_sec = results['search_insights'].get('training_time_seconds', 0)
            
            print(f"{model_name:<12} {f1:<12.4f} {test_accuracy:<12.4f} {cv_score:<12.4f}  {time_sec:<12.1f}")
        
        print("-" * 65)

        best_model, best_result = model_items[0]
        print(f"\n The overall best model: {best_model} has f1 score: ({best_result['f1_score']:.4f}")
    
    def get_model(self, model_name):
        """
        Get a trained model by name.
        
        Args:
            model_name (str): Name of model to retrieve
            
        Returns:
            sklearn model: Trained model instance
        """
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model {model_name} not found. Available: {available}")
        
        return self.models[model_name]['model']
    
    def get_model_results(self, model_name):
        """
        Get full results for a model (including search insights).
        
        Args:
            model_name (str): Name of model
            
        Returns:
            dict: Complete model results
        """
        if model_name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model {model_name} not found. Available: {available}")
        
        return self.models[model_name]
    
    def get_feature_names(self):

        """Get list of feature names used in modeling."""
        return self.feature_names
    
    def get_class_names(self):
        """Get list of class names for target variable."""
        return self.config.class_names
    
    def _count_grid_combinations(self, param_grid):
        """Count total parameter combinations in grid."""
        count = 1
        for param_values in param_grid.values():
            count *= len(param_values)
        return count

    # Model Evaluation functions

    def get_confusion_matrix(self,model_name):
        results = self.get_model_results(model_name)
        y_pred = results['predictions']
        cm = confusion_matrix(self.y_test, y_pred)
        return cm
    
    def create_confusion_plot_from_matrix(self, model_name, cm):
        fig = plt.figure(figsize=(4, 3))  
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Regular', 'Premium'], 
            yticklabels=['Regular', 'Premium'])
        plt.title(f'Confusion Matrix For: {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        return fig
    

    def create_detailed_results_table(self):
        results_table = pd.DataFrame()
        for model_name in self.models:
            # grab the model results, and from within that the classification report
            results = self.get_model_results(model_name)
            report = results['class_report']
            
            results_table[model_name] = pd.Series({
                'accuracy': results['test_accuracy'],
                # the weighted stuff is under 'weighted avg' in the classification report dict
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score'],
                # Individual classes stats are under each class label. I want premium (class 1)
                'premium_precision': report['1']['precision'],
                'premium_recall': report['1']['recall'],
                'regular_precision': report['0']['precision'],
                'regular_recall': report['0']['recall'],
                'premium_f1': report['1']['f1-score']
            })
            
        # return the transpose so that each model is a row.
        return results_table.T
    


    def plot_metrics_comparison(self, results_df, metrics_to_plot, title="Model Comparison"):
        """Plot some of a data frame column"""

        # Use basic dataframe column plotting 
        ax = results_df[metrics_to_plot].plot(kind='bar', figsize=(10, 6), title=title)
        ax.set_ylabel('Score')
        ax.set_xlabel('Models')
        ax.legend(title='Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
        return ax.figure
    
    from sklearn.metrics import roc_curve, auc

    def plot_roc_curves(self,models_to_plot=None):
        """Plot ROC curves for all models that have been run"""

        # reset things
        plt.figure(figsize=(8, 6))

        # if no models passed in, plot all of them
        if models_to_plot is None:
            models_to_plot=self.models

        for model_name in models_to_plot:
            # grab the model results, and from within that the classificaton report
            results = self.get_model_results(model_name)
            y_prob = results['probabilities']  # Get the probabilities stored for the premium class

            # The coordinates are the false positive rate, true positive rate
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)

            # Find also the AUC, summing over them all
            roc_auc = auc(fpr, tpr)
            
            # Plot the curves and include the AUC in the label.
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Best Models')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


