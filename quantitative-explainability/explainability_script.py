import sys
import os
import pandas as pd
import numpy as np
import shap
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import r2_score
#from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# PREPROCESSING

dt = pd.read_csv("quantitative-explainability/heart.csv")
dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

dt['sex'] = dt['sex'].replace("0","female")
dt['sex'] = dt['sex'].replace("1","male")

dt['chest_pain_type']=dt['chest_pain_type'].replace('1', 'typical angina')
dt['chest_pain_type']=dt['chest_pain_type'].replace('2', 'atypical angina')
dt['chest_pain_type']=dt['chest_pain_type'].replace('3', 'non-anginal pain')
dt['chest_pain_type']=dt['chest_pain_type'].replace('4', 'asymptomatic')

dt['fasting_blood_sugar']=dt['fasting_blood_sugar'].replace('0', 'lower than 120mg/ml')
dt['fasting_blood_sugar']=dt['fasting_blood_sugar'].replace('1',  'greater than 120mg/ml')

dt['rest_ecg']=dt['rest_ecg'].replace('0','normal')
dt['rest_ecg']=dt['rest_ecg'].replace('1', 'ST-T wave abnormality')
dt['rest_ecg']=dt['rest_ecg'].replace('2', 'left ventricular hypertrophy')

dt['exercise_induced_angina']=dt['exercise_induced_angina'].replace('0', 'no')
dt['exercise_induced_angina']=dt['exercise_induced_angina'].replace('1', 'yes')

dt['st_slope']=dt['st_slope'].replace('1', 'upsloping')
dt['st_slope']=dt['st_slope'].replace('2','flat')
dt['st_slope']=dt['st_slope'].replace('3','downsloping')

dt['thalassemia']=dt['thalassemia'].replace('1', 'normal')
dt['thalassemia']=dt['thalassemia'].replace('2','fixed defect')
dt['thalassemia']=dt['thalassemia'].replace('3', 'reversable defect')

dt['sex'] = dt['sex'].astype('object')
dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')
dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')
dt['rest_ecg'] = dt['rest_ecg'].astype('object')
dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')
dt['st_slope'] = dt['st_slope'].astype('object')
dt['thalassemia'] = dt['thalassemia'].astype('object')

dt = pd.get_dummies(dt, drop_first=True)

dt = dt[['age', 'resting_blood_pressure', 'cholesterol',
       'max_heart_rate_achieved', 'st_depression', 'num_major_vessels',
       'sex_1', 'chest_pain_type_1', 'chest_pain_type_2',
       'chest_pain_type_3', 'fasting_blood_sugar_1', 'rest_ecg_1',
       'rest_ecg_2', 'exercise_induced_angina_1', 'st_slope_1', 'st_slope_2',
       'thalassemia_1', 'thalassemia_2', 'thalassemia_3', 'target']]

dt.to_csv('quantitative-explainability/heart_2.csv')

def create_surrogate(X, predictions, sur_parameters):
    # Create surrogate xgboost model
    surrogate = XGBClassifier(**sur_parameters)
    surrogate.fit(X, predictions)
    sur_predictions = surrogate.predict(X)
    external_fidelity = r2_score(predictions, sur_predictions)

    return surrogate, external_fidelity

def calc_shap(model, x_explain, approximate=False):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_explain, approximate=approximate)
    return shap_values, explainer


from sklearn.neural_network import MLPClassifier
# train original model and get its predictions
X = dt.iloc[:, :-1].values
y = dt.iloc[:, -1].values
# X, y = load_breast_cancer(return_X_y=True)
X_train, _, y_train, _ = train_test_split(X, y, random_state=42)
original_model = MLPClassifier()
original_model.fit(X_train, y_train)
predictions = original_model.predict(X)

# Use agnostic explainer
sur_parameters = {'n_estimators': 100, 'max_depth': 8}
surrogate_model, external_fidelity = create_surrogate(X,
                                                    predictions,
                                                    sur_parameters)
# get the shapley values for the whole dataset
shap_values, explainer = calc_shap(surrogate_model, X)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0, :], X[0])

# summarize the effects of all the features
shap.summary_plot(shap_values, X)

# Extraction of features/explanations
# Mutliple methods

# ----------- Method 1 -----------
vals= np.abs(shap_values).mean(0)

X_train = pd.DataFrame(X_train)

feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['feature','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
feature_importance["feature"] = dt.columns[feature_importance["feature"]]

feature_importance.to_csv('quantitative-explainability/feature_importance.csv') 

# Method 2
vals = np.abs(shap_values[1])

X_train = pd.DataFrame(X_train)

feature_importance2 = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['feature','feature_importance_vals'])
feature_importance2.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
feature_importance2["feature"] =  'Feature ' + feature_importance2["feature"].astype(str)

feature_importance2.head()

# Method 3
X_train = pd.DataFrame(X_train)

feature_importance3 = pd.DataFrame(list(zip(X_train.columns, sum(shap_values))), columns=['feature','feature_importance_vals'])
feature_importance3["feature_importance_vals"] = np.abs(feature_importance3["feature_importance_vals"])

feature_importance3.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
feature_importance3["feature"] =  'Feature ' + feature_importance3["feature"].astype(str)

feature_importance3.head()

# How to view the sorted explanations
print("\n",feature_importance.head().to_string(index=False),"\n")

# Pre-process explanations for evaluation in order to explain the explanations in Human friendly way

# Reset the index of our df 
feature_importance = feature_importance.reset_index(drop=True)

############## CLEAN UP IMPORTANT EXPLANATIONS ##############

# Calculate the mean value of feature_importance_vals
z = feature_importance["feature_importance_vals"].mean()

# Remove the less important features that contributed to the decision (and round the rest)
feature_importance = feature_importance[feature_importance.feature_importance_vals >= z]
feature_importance['feature_importance_vals'] = feature_importance['feature_importance_vals'].round(+3)

feature_importance.to_csv('quantitative-explainability/feature_importance_2.csv') 
    
# QUANTITATIVE EXPLAINABILITY
    
# Calculate the mean value of the important features
y = feature_importance["feature_importance_vals"].mean()
y

# Attempt to explain depending on the situtation which features contributed to the result and how (quantitavely)
if feature_importance["feature_importance_vals"].iloc[0] >= 2 * feature_importance["feature_importance_vals"].iloc[1]:
    temp = feature_importance["feature_importance_vals"].iloc[0].astype('str')
    #flag=1 
    #while flag=1:
    #    // create temp variables
    #    // if iloc[1,2,3,4...]>= new mean
    print(">> " + feature_importance["feature"].iloc[0] + " is the most important feature in the decision making, with feature importance: " + temp)
    print(">> The rest of the contributing factors are: ")
else:
    print(">> There are many factors that contributed to the prediction. The most important are: " + feature_importance["feature"].iloc[0] + ", "
          + feature_importance["feature"].iloc[1] + " and " + feature_importance["feature"].iloc[2])
