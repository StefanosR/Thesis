import datacompy, pandas as pd

df1 = pd.read_csv(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\df1.csv')
df2 = pd.read_csv(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\df2.csv')

# C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\feature_importance_1.csv

compare = datacompy.Compare(df1,df2,
on_index=True,  #You can also specify a list of columns eg ['policyID','statecode']
abs_tol=0, #Optional, defaults to 0
rel_tol=1, #Optional, defaults to 0
df1_name='Original', #Optional, defaults to 'df1'
df2_name='New' #Optional, defaults to 'df2'
)

print(compare.report())
 