# Structure: Train MLPClassifier (non deterministic) with the make_classification dataset
# Explainability: Local LIME values with LimeTabularExplainer
# Script: Run 10 experiments, keep all feature importance values and measure 
# and measure the Mean of the ranking deviation + average std. deviation of the explanations
# Runtime: 8 minutes

from sklearn.datasets import make_classification
import numpy as np
import lime
import pandas as pd
import numpy as np
import lime.lime_tabular
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sys
import os
import statistics

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# These 2 lines are for redirecting the python output to a log file
stdoutOrigin=sys.stdout
sys.stdout = open("log_LEL_S1_MLPC_ND_BD.txt", "w")

#------------------------------------------------------------------------------------------------------------------------------------------------

def myscript(iteration_number, number_features):
        
    X, y = make_classification(n_samples=50000, n_features=30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = MLPClassifier()
    model.fit(X_train, y_train)

    z = pd.DataFrame(X)

    feature_names = []
    for i in z.columns:
        feature_names.append("Feature "+str(i))

    categorical_features = np.argwhere(np.array([len(set(X_train[:,x])) for x in range(X_train.shape[1])]) <=10).flatten()

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, 
                                                    categorical_features = categorical_features, verbose=True, mode='classification')

    # Choose which instance to examine
    input = 24

    # Produce explanations for the instance
    exp = explainer.explain_instance(X[input], model.predict_proba, num_features=number_features)

    # # Visualized explanations
    # exp.show_in_notebook(show_table=True)

    # the explanations list contains the following:
    # 1. The feature contribution (weight) of all the features (ranked by absolute value)
    # 2. The explanation of each feature contribution (how the feature's numerical value affects the prediction)
    explanations = exp.as_list()

    features = feature_names

    # Extract LIME explanations
    feature_importance = []
    for i in range(len(explanations)):
        for word in features:
        # if a word from the features list is present in the explanation strings (1st element of the list) then grab it
            if word+' ' in explanations[i][0]:
                # transfer the feature and its value to a 
                feature_importance.append("%s = " %word + str("{:.3f}".format(abs(explanations[i][1]))))

    # Save explanations of each run to a temp file         
    with open(r"C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\L_E_LIME\features.txt", "a") as file:
            file.write("Run number: %d\n\n" %iteration_number)
            for row in feature_importance:
                s = "".join(map(str, row))
                file.write(s+'\n')
            file.write('\n')
    file.close()
    
    # ----------- Extract local predictions through the loops -----------
    def grab_predictions(instance,iteration):
        prediction = model.predict(instance)
        preds = []
        if prediction.item(0) == 1:
            preds.append("Iteration Number %d: " %iteration +"Breast cancer detected :(")
        else:
            preds.append("Iteration Number %d: " %iteration +"You are okay :)")
        preds = pd.DataFrame(preds)  
        preds.to_csv(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\L_E_LIME\predictions.txt', header=None, index=None, sep=' ', mode='a')
    
    local = X[input]
    from numpy import asarray
    instance = asarray([local])
    grab_predictions(instance, iteration_number) # This will show us how the prediction changes for the instance. Will be interesting to compare with 
    # the input perturbation script to see if it also affects predictions alongside explanations
    
# ------------------------------------------------------------------------------------------------------------------------------------------------

# Runs the Script (repeat N times the experiment)
experiments = 10

# Control the number of features we want to examine (top 5)
number_features = 5

def main(unused_command_line_args):
    for i in range(experiments):
        myscript(i+1, number_features)
    return 0

main(sys.argv)

# Extract the explanations from the different runs

with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\L_E_LIME\features.txt','r') as file:
    data = file.readlines()

file.close()

#-----------------------------------------------------------------------------------------------------------------
    
# Make script that compares the  data - explanations, and the highlighted features (if top feature is the same)

# This is a function for getting the change % of a feature (will not be used here)
def get_change(current, previous):
    
    if current == previous:
        return 100.0
    try:
        change = (abs(current - previous) / ((current + previous)/2) ) * 100.0
        return "{:.2f}".format(change)
    except ZeroDivisionError:
        return 0

values = []
features = []
for i in range(experiments):
    for j in range(number_features):
        x = data[i*8+j+2].split()
        values.append(x[3])
        features.append(x[0]+x[1])

# Make changes here on the dimensions of the dataframes for evaluating more experiments (here we have 5 features)
  
# Make dataframe with the weights of the features for each run  
values = np.array(values)
sub_arrays = np.hsplit(values, experiments)
df = pd.DataFrame(sub_arrays)
for i in range(experiments):
    df.index = ['Run_%d'%(i+1) for i in range(experiments)]
df.columns = ['1st Feature', '2nd Feature', '3rd Feature', '4th Feature', '5th Feature']

# Make dataframe with the feature names (ranking of the features at each experiment)
features = np.array(features)
sub_arrays2 = np.hsplit(features, experiments)
df2 = pd.DataFrame(sub_arrays2)
for i in range(experiments):
    df2.index = ['Run_%d'%(i+1) for i in range(experiments)]
df2.columns = ['1st Feature', '2nd Feature', '3rd Feature', '4th Feature', '5th Feature']
df2 = df2.applymap(lambda x: x.replace('=', ''))

# Measure the difference of the 1st feature between the different runs (if the top feature is the same)

# if df2.loc["Run_1","1st Feature"] == df2.loc["Run_2","1st Feature"]:
#     print(get_change(float(df.loc["Run_1","1st Feature"]),float(df.loc["Run_2","1st Feature"])),'%')
# else:
#     print(df2)

# Gather here the unique features
uniqueValues = (df2['1st Feature'].append(df2['2nd Feature'].append(df2['3rd Feature'].append(df2['4th Feature'].append(df2['5th Feature']))))).unique()

df2 = df2.astype(str)
print("\nOur dataframes are:\n\n",df,"\n\n",df2,"\n")  
print("\nDescribe df with the feature names:\n\n", df2.describe())
print("\n..........................................\n")    

#-----------------------------------------------------------------------------------------------------------------

# Make a dictionary for storing the unique feature names as keys and their corresponding weights as values

df1 = df.astype(float)
dict = {animal: df1[df2.eq(animal)].stack().values for animal in np.unique(df2)}
print("\nThis is the new dictionary we created:\n", dict)

# Calculate std deviation for each feature

matrix = []
matrix2 = []
matrix3= []
matrix4 = []

# In matrix we will store the keys (features) and in matrix2 the values (weights)
for i in dict:
    matrix.append(i)
    matrix2.append(dict[i])

# In matrix3 we calculate the standard deviation of the features
for alist in matrix2:
    matrix3.append("{:.2f}".format(np.std(alist, dtype=np.float64)))

# In matrix4 we calculate the relative standard deviation (percentage)
for alist in matrix2:
    matrix4.append(str("{:.2f}".format(100*np.std(alist, dtype=np.float64)/statistics.mean(alist)))+"%")
    
print("\nThis is the matrix with the unique features: \n", *matrix, sep = '\n')
print("\nThis is the matrix with the list of values each feature has: \n", *matrix2, sep = '\n')   
print("\nThis is the the std deviation of each list: \n", *matrix3, sep = '\n')  
print("\nThis is the relative std deviation of each list of values: \n", *matrix4, sep = '\n')

# Function for returning a list of the keys of a dictionary

def getList(dict):
    feature_names = []
    for key in dict.keys():
        feature_names.append(key)
          
    return feature_names

keys = getList(dict) 

# Concatenate the lists of the features and the deviations
std_deviation_of_features = list(zip(keys, matrix3))
re_std_deviation_of_features = list(zip(keys, matrix4))

print("\nThe standard deviation of our features is: \n", *std_deviation_of_features, sep = '\n')
print("\nThe relative standard deviation of our features is: \n", *re_std_deviation_of_features, sep = '\n')
print("\n..........................................\n")    

#-----------------------------------------------------------------------------------------------------------------

# Measuring the ranking deviation of the features

# Function returning the index positions of a df where an element is present
def getIndexes(dfObj, value):
     
    # Empty list
    listOfPos = []
     
    # isin() method will return a dataframe with
    # boolean values, True at the positions   
    # where element exists
    result = dfObj.isin([value])
     
    # any() method will return
    # a boolean series
    seriesObj = result.any()
 
    # Get list of column names where
    # element exists
    columnNames = list(seriesObj[seriesObj == True].index)
    
    # Iterate over the list of columns and
    # extract the row index where element exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
 
        for row in rows:
            listOfPos.append((row, col))
             
    # This list contains a list tuples with
    # the index of element in the dataframe
    return listOfPos
 
# Calling getIndexes() function 

listOfPositions = []
for item in uniqueValues:
    listOfPositions.append(str(item) + str(getIndexes(df2, item)))   
    
print("\nThese are the positions where each feature is present and what ranking it has:\n ",*listOfPositions, sep = '\n')  

# In how many columns the feature occurs
# len(listOfPositions[5].split(', ')) / 2

listOfPositions2 = {}
for item in uniqueValues:
    listOfPositions2[item] = getIndexes(df2, item)
    
# for item in uniqueValues:
#     print(listOfPositions2["%s"%item])

# c = df2.columns[(df2 == 'Feature23').any()].tolist()
# d = df2.value_counts()

# matrix = []
# for col in df2.columns:
#     matrix.append(df2[col].value_counts())

# Magic matrix contains a list of integers reflecting the positions where each feature is present in the ranking
magic = []
keys = []
rankings = {}
for item in uniqueValues:
    new = []
    for col in df2.columns:
        for row in df2.index:
            if df2.at[row,col] == item:
                new.append(int(str(col).split(' ')[0][0]))
    magic.append(new)

# The rankings array will contain the std deviation of the rankings measured before
rankings = []
for alist in magic:
    rankings.append(np.std(alist, dtype=np.float64))

# The ranking_deviation concatenates the unique features with the std deviation into a comprehensive list
ranking_deviation = []
for i in range(len(uniqueValues)):
    ranking_deviation.append(uniqueValues[i]+": "+str(rankings[i]))

print("\nThe ranking deviation of each feature is: \n", *ranking_deviation, sep = '\n')

rankings_percentage = []
for alist in magic:
    rankings_percentage.append(str("{:.2f}".format(100*np.std(alist, dtype=np.float64)/statistics.mean(alist)))+"%")
    
ranking_deviation_percentage = []
for i in range(len(uniqueValues)):
    ranking_deviation_percentage.append(uniqueValues[i]+": "+str(rankings_percentage[i]))   
    
print("\nThis is the ranking deviation percentage for each feature: \n",*ranking_deviation_percentage, sep = '\n')
print("\n..........................................\n")    
    
# Research ranking deviation and visualize other statistics (distribution)   
# Expand on more runs and features (also on scenario 2)

# # Save our dataframes for reference
# df1.to_csv(r'df1.csv', index = False)
# df2.to_csv(r'df2.csv', index = False)

# Calculate the mean of the ranking deviation

rankings_percentage_float = []
for alist in magic:
    rankings_percentage_float.append(float("{:.2f}".format(100*np.std(alist, dtype=np.float64)/statistics.mean(alist))))
    
final_result = str("{:.2f}".format(np.mean(rankings_percentage_float)))

# Print the mean of the ranking deviation
print("Mean of the ranking deviation: "+final_result+"%")

# Calculate the mean of the std deviation
std_deviation_float = []
for alist in matrix2:
    std_deviation_float.append(float("{:.2f}".format(100*np.std(alist, dtype=np.float64)/statistics.mean(alist))))
    
final_result2 = str("{:.2f}".format(np.mean(std_deviation_float)))

# Print the mean of the std deviation
print("\nMean of the standard deviation (from importance of features): "+final_result2+"%")

# -----------------------------------------------------------------------------------------------------------------------------

# Check if all predictions where equal during these runs

with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\L_E_LIME\predictions.txt','r') as file:
    data = file.readlines()
file.close()

predz = []
for i in range(experiments):
        x = data[i].split(':')
        predz.append(x[1])
        
def all_equal_ivo(lst):
    return not lst or lst.count(lst[0]) == len(lst)

if all_equal_ivo(predz) == True:
    print("\nAll predictions were the same")
else:
    print("\nNot all predictions were the same")

# print("\n",*data,sep="")

# -----------------------------------------------------------------------------------------------------------------------------    

# The next 2 lines are for saving the python script output in a log file
sys.stdout.close()
sys.stdout=stdoutOrigin

os.remove(r"C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\L_E_LIME\features.txt")
os.remove(r"C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\L_E_LIME\predictions.txt")