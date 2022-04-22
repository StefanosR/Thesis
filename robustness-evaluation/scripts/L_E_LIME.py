# Structure: Train MLPClassifier (non deterministic) with the load_breast_cancer dataset
# Explainability: Global SHAP values with KernelExplainer
# Script: Run 10 experiments, keep all feature importance values and measure 
# and measure the Mean of the ranking deviation + average std. deviation of the explanations
# Runtime: 4 hours for X, 1 hour for X_test

import sys
import pandas as pd
import numpy as np
import shap
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os