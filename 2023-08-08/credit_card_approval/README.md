# Credit card approval
This directory contains an example pipeline going from dataset cleaning to model training and evaluation. 
It implements a simplistic regression task using the [credit approval dataset](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) from Kaggle.
Based on users attributes and credit history, we want to predict for new users the average number of days past due date 
they will pay off their loan. The functions are implemented in `data_cleaning.py`, `features.py`, and `model_training.py`,
and are executed through `credit_card_approval.ipynb`.