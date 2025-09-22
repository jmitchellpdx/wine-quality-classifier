# src/config.py
"""
Configuration class for wine quality classification project.
Centralizes all parameters to avoid hard-coding values in pipeline.
"""
import os

class WineConfig:
    """
    Simple container for configuration data ModelPipeline. 
    The parameters for each model are in two dictionaries:
        - model_base_params are for parameters that will not change over the grid search
        - model_grid_params are the grid search that will be used by GridSearchCV
    """
    
    def __init__(self):
        # Data file paths
        self.raw_data_path = 'data/raw/winequality-all.csv'
        self.processed_data_path = 'data/processed/wine_unique.csv'
        
        # Target variable configuration: convert values from 3 to 9 to binary classification
        self.target_col = 'quality'
        self.class_names = ['regular', 'premium']
        self.regular_scores = [3, 4, 5, 6]    # regular wine quality scores
        self.premium_scores = [7, 8, 9]       # premium wine quality scores
        
        # Data splitting and randomization
        self.test_size = 0.2
        self.cv_folds = 5
        self.random_state = 42

         # Feature preprocessing configuration
        self.scale_features = True
        # Whether to apply additional transformations (such as log scaling)
        self.apply_custom_transformations = True
        # Features identified in EDA as highly skewed - apply log transform before scaling
        self.features_to_scale = ['chlorides','sulphates','residual sugar','fixed acidity','volatile acidity','free sulfur dioxide']
     

        # whether to drop less important features
        # All the features:  ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'wine_type_binary']
        self.drop_features=False
        # This was the least important feature in random forest. 
        self.features_to_drop = ['wine_type_binary']

       
        # Use f1_weighted as primary metric due to class imbalance between regular and premium
        self.scoring_metric = 'f1_weighted'  

        # Changing level of detail
        self.pipeline_verbose=True # whether to show additional detail for pipeline model searches
        self.grid_search_verbosity=1  # use 1 for regular verbosity, 2 for extra detail
        
        # Models to train and evaluate. 
        # To run one model: self.models_to_run = ['mlp']

        self.models_to_run = ['rf', 'svm', 'logistic', 'mlp']

        
        # Base parameters that stay constant during grid search
        self.model_base_params = {
            'logistic': {
                'penalty': 'l2',
                'max_iter': 1000,
                'solver': 'saga',  ### If doesn't work well, try 'lbfgs'
                'class_weight': 'balanced',
                'n_jobs': -1,  ###use all available cores for parallel processing
                'random_state': self.random_state
            },
            'rf': {
                'class_weight': 'balanced',
                'n_jobs': -1,  ###use all available cores for parallel processing
                'random_state': self.random_state
            },
            'svm': {
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'probability': True,   # This stores probabilities of class labels as well as predictions
                'random_state': self.random_state
                # Note: svm doesn't support n_jobs
            },
            'mlp': {
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1,  # This is the fraction during training to use to avoid overfitting. 
                'n_iter_no_change': 20,  # Patience parameter - how long to keep going with no improvement
                'random_state': self.random_state
                # Note: MLP doesn't support class_weight or n_jobs
            }
        }

    
        self.model_grid_params = {
            # Version 4: Just the best models (no grid!!)
            'logistic': {
                'C': [0.8]  # Regularization strength - higher C means less regularization

            },
            'rf': {
                'n_estimators': [55], # number of decision trees in the ensemble
                'max_depth': [23],  # Maximum tree depth (controls complexity)
                'max_features': ['sqrt'],  # Features considered at each split (square root of total number)
                'min_samples_split': [10],  # the minimum number of samples required to split a node
                'min_samples_leaf': [3]    #  the minimum samples in each leaf node
            },
            'svm': {
                'C': [2], # Regularization parameter
                'gamma': [0.8] # RBF kernel width - higher gamma = more complex decision boundary
            },
            'mlp': {
                'solver': ['sgd'], # Stochastic gradient descent
                'hidden_layer_sizes': [(120,60)], # 2 hidden layers: 120 then 60 neurons
                'alpha': [0.1], # L2 regularization strength
                'learning_rate_init': [0.02], # Initial learning rate
                'momentum': [0.95] # Momentum for gradient descent
            }
        }

    # Get the list of models that will be run
    def get_models_to_run(self):
        return self.models_to_run
    
    # get the base parameters for each model
    def get_base_params(self, model_name):
        if model_name not in self.model_base_params:
            raise ValueError(f"No parameter grid defined for model: {model_name}")
        return self.model_base_params[model_name]  
    
    # get the hyperparameter grid
    def get_grid_params(self, model_name):
        if model_name not in self.model_grid_params:
            raise ValueError(f"No parameter grid defined for model: {model_name}")
        return self.model_grid_params[model_name]
    
    
    def validate_config(self):
        """
        Check that all requested models have the parameters they need.
        """
        print("VALIDATING CONFIGURATION")
        for model in self.models_to_run:
            if model not in self.model_base_params:
                raise ValueError(f"Model '{model}' has no base parameters defined")
            if model not in self.model_grid_params:
                raise ValueError(f"Model '{model}' has no hyperparameter grid defined")
        print("....Config looks fine!")
    

