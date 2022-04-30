# Structure: LGBMRegressor (non deterministic) with the diabetes dataset
# Explainability: Global SHAP values with TreeExplainer
# Script: Run 10 experiments, keep all feature importance values and measure 
# and measure the Mean of the ranking deviation + average std. deviation of the explanations

from cmath import exp
import sys
import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
import os

# These 2 lines are for redirecting the python output to a log file
stdoutOrigin=sys.stdout 
sys.stdout = open("log_GES_S1_1_ND_MR.txt", "w")

# ------------------------------------------------------------------------------------------------------------------------------------------------

def myscript(iteration_number):
    
    print("Iteration:", iteration_number)
    
    # Train model
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    # Calculate shap values 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # ----------- Extract global explanations -----------
    feature_names = load_diabetes().feature_names
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    vals = np.abs(shap_df.values).mean(0) # get the mean of the shap values for each feature
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

    shap_importance["col_name"] =  'Feature ' + shap_importance["col_name"].astype(str)
    feature_importance = shap_importance.head()

    feature_importance.to_csv(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_SHAP\features.txt', header=None, index=None, sep=' ', mode='a')

    with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_SHAP\features.txt','a') as file:
        file.write("\nRun number: 1\n\n")
    file.close()
    
# ------------------------------------------------------------------------------------------------------------------------------------------------

# Runs the Script (repeat N times the experiment)

experiments = 10

def main(unused_command_line_args):
    for i in range(experiments):
        myscript(i+1)
    return 0

main(sys.argv)
    
with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_SHAP\features.txt','r') as file:
    data = file.readlines()

j=0
for i in range(6,8*experiments,8):
    j+=1
    data[i] = 'Run number: '+str(j+1)+'\n'

# Fix the names of the runs
with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_SHAP\features.txt','w') as file:
    file.writelines( data )
    
file.close()

# Extract the explanations from the different runs

with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_SHAP\features.txt','r') as file:
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
    for j in range(5):
        x = data[i*8+j].split()
        values.append(x[2])
        features.append(x[0]+x[1])

# Make changes here on the dimensions of the dataframes for evaluating more experiments
        
# Make dataframe with the shapley values of the features for each run  
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
df2 = df2.applymap(lambda x: x.replace('"', ''))

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

# Make a dictionary for storing the unique feature names as keys and their corresponding shapley values as values

df1 = df.astype(float)
dict = {animal: df1[df2.eq(animal)].stack().values for animal in np.unique(df2)}
print("\nThis is the new dictionary we created:\n", dict)

# Calculate std deviation for each feature

matrix = []
matrix2 = []
matrix3= []
matrix4 = []

# In matrix we will store the keys (features) and in matrix2 the values (shapley values)
for i in dict:
    matrix.append(i)
    matrix2.append(dict[i])

# In matrix3 we calculate the standard deviation of the features 
for list in matrix2:
    matrix3.append("{:.2f}".format(np.std(list, dtype=np.float64)))

# In matrix4 we calculate the relative standard deviation (percentage)
for list in matrix2:
    matrix4.append(str("{:.2f}".format(100*np.std(list, dtype=np.float64)/len(list)))+"%")
    
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

del list

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
for list in magic:
    rankings.append(np.std(list, dtype=np.float64))

# The ranking_deviation concatenates the unique features with the std deviation into a comprehensive list
ranking_deviation = []
for i in range(len(uniqueValues)):
    ranking_deviation.append(uniqueValues[i]+": "+str(rankings[i]))

print("\nThe ranking deviation of each feature is: \n", *ranking_deviation, sep = '\n')

rankings_percentage = []
for list in magic:
    rankings_percentage.append(str("{:.2f}".format(100*np.std(list, dtype=np.float64)/len(list)))+"%")
    
ranking_deviation_percentage = []
for i in range(len(uniqueValues)):
    ranking_deviation_percentage.append(uniqueValues[i]+": "+str(rankings_percentage[i]))   
    
print("\nThis is the ranking deviation percentage for each feature: \n",*ranking_deviation_percentage, sep = '\n')
print("\n..........................................\n")    
    
# Research ranking deviation and visualize other statistics (distribution)   
# Expand on more runs and features (also on scenario 2)

# Save our dataframes for reference
df1.to_csv(r'df1.csv', index = False)
df2.to_csv(r'df2.csv', index = False)

# Calculate the mean of the ranking deviation

rankings_percentage_float = []
for alist in magic:
    rankings_percentage_float.append(float("{:.2f}".format(100*np.std(alist, dtype=np.float64)/len(alist))))
    
final_result = str("{:.2f}".format(np.mean(rankings_percentage_float)))

# Print the mean of the ranking deviation
print("Mean of the ranking deviation: "+final_result+"%")

# Calculate the mean of the std deviation
std_deviation_float = []
for alist in matrix2:
    std_deviation_float.append(float("{:.2f}".format(100*np.std(alist, dtype=np.float64)/len(alist))))
    
final_result2 = str("{:.2f}".format(np.mean(std_deviation_float)))

# Print the mean of the std deviation
print("\nMean of the standard deviation (from importance of features): "+final_result2+"%")
    
# The next 2 lines are for saving the python script output in a log file
sys.stdout.close()
sys.stdout=stdoutOrigin

os.remove(r"C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_SHAP\features.txt")