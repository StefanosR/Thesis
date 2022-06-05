import gradio as gr
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import seaborn as sns

def choose_script(Explainability_method, Category, Implementation, Scenario, Dataset, Loop, Feature):
    
    import gradio as gr
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import numpy as np
    import seaborn as sns
    plt.style.use('seaborn')
    plt.rcParams["figure.figsize"] = (8,4)
    global weights
    
    method = Explainability_method
    category = Category
    implementation = Implementation
    scenario = Scenario
    dataset = Dataset
    loop = Loop
    
    if implementation == "LGBM":
        i = "LGBM"
    elif implementation == "MLPC":
        i = "MLPC"
    else:
        print("Invalid implementation")  
          
    if category == "Global":
        c = "GE"
    elif category == "Local":
        c = "LE"
    else:
        print("Invalid category")
    
    if method == "LIME":
        m = "L"
    elif method == "SHAP":
        m = "S"
    elif method == "MASHAP":
        m = "M"
    else:
        print("Invalid method")

    if scenario == "1st":
        s = "S1"
        det = "ND"
    elif scenario == "2nd":
        s = "S2"
        det = "ND"
    elif scenario == "3rd":
        s = "S3"
        det = "D"
    else:
        print("Invalid scenario")
        
    if dataset == "small":
        d = "SD"
    elif dataset == "big":
        d = "BD"
    else:
        print("Invalid dataset")
    
    if loop == "1st":
        l = "_1L"
    elif loop == "2nd":
        l = "_2L"
    elif loop == "3rd":  
        l = "_3L"
    else:
        l = ""

    search_word1 = 'Mean of the standard deviation'
    search_word2 = 'Mean of the ranking deviation'

    log = "log_%s%s_%s_%s_%s_%s%s.txt" % (c, m, s, i, det, d, l)
    feature = Feature
    
    import matplotlib.image as mpimg
    img = mpimg.imread('plots/warning.png')
        
    if (m == "S" and d == "BD" and i == "MLPC"):
        return "Warning: SHAP crashes with big datasets on MLPC","Please change inputs", img
    elif (m == "L" and c == "GE" and i == "LGBM"):
        return "Warning: LIME global explanations are not supported for regression","Please change inputs", img
    elif ((s == "S2" or s == "S3") and l == ""):    
        return "Warning: Scenarios 2 and 3 require a loop choice to present results","Please change inputs", img
    elif (s == "S2" and l == "_3L"):    
        return "Warning: Scenario 2 does not have a 3rd loop","Please change inputs", img
    elif (s == "S1" and l != ""):    
        return "Warning: Scenario 1 does not have any loops","Please change inputs", img
    else:
        img = mpimg.imread('plots/error.png')
        try:
            if s == "S3":
                weights = []
                for j in range(2):
                    search_word = 'Run_%s ' % (j+1)
                    with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\experiments\%s'% (log)) as file:  
                        line1 = [line1 for line1 in file.read().split('\n') if search_word in line1]
                    weights.append((line1[0].split())[feature])         
            else:
                weights = []
                for j in range(10):
                    search_word = 'Run_%s ' % (j+1)
                    with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\experiments\%s'% (log)) as file:  
                        line1 = [line1 for line1 in file.read().split('\n') if search_word in line1]
                    weights.append((line1[0].split())[feature]) 
        except:
            return "Something went wrong on the weights calculation", "Please try another input", img
        
        try:
            floats = [float(x) for x in weights]
        except ValueError:
            return "Something went wrong on the floats conversion", "Please try another input", img
        
        # plt.style.use('seaborn-whitegrid')
        plt.style.use('seaborn')
        plt.rcParams["figure.figsize"] = (12.8,6.8)
        sns.histplot(floats, bins=5, kde=True, stat="count").set(xlabel='Values', title='Distribution of the weights for Feature #%d' % (feature))
        sns.set(font_scale = 1.5)
        plt.tight_layout()
        plt.savefig('plots/distribution_%s_%s_%s_%s_%s_%sfeature#%d.png' % (c, m, s, i, d, l, feature), bbox_inches='tight', format='png')
        img = mpimg.imread('plots/distribution_%s_%s_%s_%s_%s_%sfeature#%d.png' % (c, m, s, i, d, l, feature))
        plt.clf()

        # opens it, then closes it automatically at the end of the with block
        with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\experiments\%s'% (log)) as file:  
            line1 = [line1 for line1 in file.read().split('\n') if search_word1 in line1]
        with open(r'C:\Users\Stefanos Rigas\Documents\Projects\Thesis\thesis\experiments\%s'% (log)) as file: 
            line2 = [line2 for line2 in file.read().split('\n') if search_word2 in line2]
        return ((str(line1).strip("['']")).split(":"))[1], ((str(line2).strip("['']")).split(":"))[1], img

def login(username,password):
    if (username=="stefanos" or "kanellopoulos" or "makris") and password=="thesis":
        return True
    elif username== "admin" and password==" ":
        return True
    else:
        return False
    
demo = gr.Interface(
fn=choose_script,
inputs=[gr.Radio(["SHAP", "MASHAP", "LIME"]),gr.Radio(["Local","Global"]),gr.Radio(["LGBM","MLPC"]),gr.Radio(["1st","2nd","3rd"]),gr.Radio(["big","small"]),gr.Radio(["none","1st","2nd","3rd"]), gr.Slider(1, 5,step=1)],
outputs=[gr.Textbox(label="Mean Coefficient of Variation (from the weights)"),gr.Textbox(label="Mean Coefficient of Variation (from the rankings)"),gr.Image(label="Image output")],
title="Robustness Evaluation of Machine Learning Explainability Methods",
thumbnail="plots/thumbnail.png",
theme="default", 
live=True,
article="<center>Stefanos Rigas - Computer Engineering and Informatics Department - University of Patras</center>",
description="""The inputs on the left allow the user to switch the settings of the experiment. The outputs on the right show the mean 
relative standard deviation (or Coefficient of Variation) of the features, meaning how the weights and the rankings of the top features 
produced by the explainability methods are distributed. Low percentage signifies a more robust system, while high percentages imply 
instability. The output image is a reflection of the distribution of the weights (from 10 runs) for the feature that was specified. 
The plot inlcudes the kernel density estimation of the feature's values. Some setting combinations will produce warnings (orange) or errors (red). 
Warnings can be simply corrected by applying the proper settings which are specified in the outputs. The errors that are identified 
by the interface are under debugging. If the interface is not working properly, flag the current instance.""",
)
demo.launch(auth=login,auth_message="Please provide your credentials",favicon_path="plots/logo.png",share=True)