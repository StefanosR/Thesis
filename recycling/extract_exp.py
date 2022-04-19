with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','r') as file:
    data = file.readlines()

data[6] = 'Run number: 2\n'
data[14] = 'Run number: 3\n'
data[22] = 'Run number: 4\n'
data[30] = 'Run number: 5\n'
data[38] = ''


# Write the new model on to_matlab.py
with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','w') as file:
    file.writelines( data )
    
file.close()

# Extract the explanations from the different runs

with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\robustness-evaluation\explanations\G_E_MASHAP\features.txt','r') as file:
    data = file.readlines()
    
# Make script that compares the  data - explanations, and the highlighted features

# if top feature is the same)

def get_change(current, previous):
    
    if current == previous:
        return 100.0
    try:
        change = (abs(current - previous) / ((current + previous)/2) ) * 100.0
        return change
    except ZeroDivisionError:
        return 0

feature1 = data[8].split()
feature2 = data[16].split()
value1 = feature1[2]
value2 = feature2[2]

if feature1[1] == feature2[1]:
    print(get_change(float(value1),float(value2)),'%')
    
    