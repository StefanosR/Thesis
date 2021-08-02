import sys
import pandas as pd
import numpy as np
import shap
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def myscript(iteration_number):

    print("Iteration:", iteration_number)
    
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
    X, y = load_breast_cancer(return_X_y=True)
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
    ## shap.force_plot(explainer.expected_value, shap_values[0, :], X[0])

    # summarize the effects of all the features
    ## shap.summary_plot(shap_values, X)

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Extraction of features/explanations (mutliple methods)
    
    # ----------- Method 1 -----------
    vals= np.abs(shap_values).mean(0)

    X_train = pd.DataFrame(X_train)

    feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance["col_name"] =  'Feature ' + feature_importance["col_name"].astype(str)

    print("Feature Importance - Method 1:", feature_importance.head())
     
    # ----------- Method 2 -----------
    vals = np.abs(shap_values[1])

    X_train = pd.DataFrame(X_train)

    feature_importance2 = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance2.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance2["col_name"] =  'Feature ' + feature_importance2["col_name"].astype(str)

    print("Feature Importance - Method 2:", feature_importance2.head())

    # ----------- Method 3 -----------
    X_train = pd.DataFrame(X_train)

    feature_importance3 = pd.DataFrame(list(zip(X_train.columns, sum(shap_values))), columns=['col_name','feature_importance_vals'])
    feature_importance3["feature_importance_vals"] = np.abs(feature_importance3["feature_importance_vals"])

    feature_importance3.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    feature_importance3["col_name"] =  'Feature ' + feature_importance3["col_name"].astype(str)

    print("Feature Importance - Method 3:", feature_importance3.head())

    # ------------------------------------------------------------------------------------------------------------------------------------------------
    
    # How to view the sorted explanations
    print("\n",feature_importance.head().to_string(index=False),"\n")

    # CHECK QE CODE FOR THIS PART
    
    # Pre-process explanations for evaluation (explain the explanations in Human friendly way)

    # Reset the index of our df 
    feature_importance = feature_importance.reset_index(drop=True)

    # # Calculate the mean value of feature_importance_vals
    # z = feature_importance["feature_importance_vals"].mean()
    # z

    # # Remove the less important features that contributed to the decision (and round the rest)
    # feature_importance = feature_importance[feature_importance.feature_importance_vals >= z]
    # feature_importance['feature_importance_vals'] = feature_importance['feature_importance_vals'].round(+3)
    # feature_importance
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    
    xfile_name = "robustness-evaluation/feature_importance_%d.csv" % iteration_number
    with open(xfile_name, "w") as xf:
        feature_importance.to_csv('robustness-evaluation/feature_importance_%d.csv' % iteration_number) 

# CHECK QE CODE FOR THIS PART     

# # Calculate the mean value of the important features
# y = feature_importance["feature_importance_vals"].mean()
# y

# # Attempt to explain depending on the situtation which features contributed to the result and how (quantitavely)
# if feature_importance["feature_importance_vals"].iloc[0] >= 2 * feature_importance["feature_importance_vals"].iloc[1]:
#     temp = feature_importance["feature_importance_vals"].iloc[0].astype('str')
#     #flag=1 
#     #while flag=1:
#     #    // create temp variables
#     #    // if iloc[1,2,3,4...]>= new mean
#     print(">> " + feature_importance["col_name"].iloc[0] + " is the most important feature in the decision making, with feature importance: " + temp)
#     print(">> The rest of the contributing factors are: ")
# else:
#     print("1. There are many factors that contributed to the prediction. The most important are:")

 # ------------------------------------------------------------------------------------------------------------------------------------------------
 
 # Runs the Script
 
def main(unused_command_line_args):
    for i in range(1):
        myscript(i+1)
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
    
# The end