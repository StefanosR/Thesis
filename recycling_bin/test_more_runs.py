import sys
import pandas as pd
import numpy as np
import shap
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#-----------------------------------------------------------------------------------------------------------------

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

    # ----------- Extract global explanations -----------
    vals= np.abs(shap_values).mean(0)

    X_train = pd.DataFrame(X_train)

    feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
    feature_importance["col_name"] =  'Feature ' + feature_importance["col_name"].astype(str)

    feature_importance = feature_importance.head()

    feature_importance.to_csv(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt', header=None, index=None, sep=' ', mode='a')

    with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','a') as file:
        file.write("\nRun number: 2\n\n")
    file.close()
    
# ------------------------------------------------------------------------------------------------------------------------------------------------

# Runs the Script (repeat N times the experiment)

experiments = 10

def main(unused_command_line_args):
    for i in range(experiments):
        myscript(i+1)
    return 0

main(sys.argv)
    
with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','r') as file:
    data = file.readlines()

j=0
for i in range(6,8*experiments,8):
    j+=1
    data[i] = 'Run number: '+str(j+1)+'\n'

# Fix the names of the runs
with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','w') as file:
    file.writelines( data )
    
file.close()

# Extract the explanations from the different runs

with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','r') as file:
    data = file.readlines()

file.close()

#-----------------------------------------------------------------------------------------------------------------