# Wine Quality Classifier - 5509 (Supervised learning) Final Project

## Overview
Binary classification of wine quality (regular vs premium) using chemical features from UCI Wine Quality Dataset. Compares 4 supervised learning models after tuning on a training set. Random Forest acheived the best overall performance with 84% accuracy. Using Bayesian methods, the model was shown to have potential as a "pre-filter" for premium wines: boosting the chance of selecting a premium wine from 19% (without pre-filtering) to 57% (with pre-filtering using the Random Forest classifier).

## File Structure
- `Report.ipynb` - **Main deliverable**: Complete analysis and results
- `EDA.ipynb` - Exploratory data analysis and preprocessing
- `Modeling_And_Evaluation.ipynb` - Model training and comparison
- `src/` - Python tools used by the notebooks (config, pipeline, utilities)
- `figures/` - Generated plots referenced in report (in .png format)
- `results/` - Model performance metrics (CSV format)
- `data/raw/` - UCI Wine Quality dataset
- `data/processed/` - Cleaned data (red and wine examples combined, duplicates removed)
- `5509_Final_Project_Presentation.mp4` Overview video of the project (11min 50sec duration)

## Models Evaluated
- Logistic Regression 
- Support Vector Machine  
- Random Forest 
- Multi-Layer Perceptron

## Usage
Start with `Report.ipynb` for complete analysis. The two other notebooks were used to generate the results in the report.

**Dataset:** UCI Wine Quality (6,497 samples, 12 chemical features)
**Task:** Binary classification into regular (low or average quality score) vs premium (above average quality score)