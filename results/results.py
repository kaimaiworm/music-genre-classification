import pathlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix, precision_score,  recall_score, f1_score, roc_auc_score, accuracy_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
# import packages for hyperparameters tuning
import optuna
from sklearn.model_selection import StratifiedKFold

from optuna.visualization.matplotlib import plot_optimization_history
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import cross_validate
import pickle
from hyperparameter_stack import objective, stack_tuning
from stacking_functions import train_oof_predictions, model_selector, create_meta_dataset, stack_prediction
from sklearn.linear_model import LogisticRegression
import scikitplot as skplt
import time
import copy
from sklearn.metrics import PrecisionRecallDisplay
from xgboost import plot_importance
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.io as pio
from plotly.subplots import make_subplots
pio.renderers.default='browser'

#import warnings
#warnings.filterwarnings('ignore')


# current working directory
path = pathlib.Path().absolute()


################
## Folders ##
################

data_input_folder    = "{}\output".format(path.__str__().replace("results", "dataset"))
base_input_folder    = "{}\output".format(path.__str__().replace("results", "base_models"))
stack_input_folder   = "{}\output".format(path.__str__().replace("results", "stacking"))   
multi_input_folder   = "{}\output".format(path.__str__().replace("results", "multiclass"))   
reduction_input_folder = "{}\output".format(path.__str__().replace("results", "reduction"))  
                                    
output_folder        = "{}\\output".format(path)


################
# Load Dataset #
################

# Data for Pop Vs Other Classification
df_train = pd.read_excel("{}\\genres_bi.xlsx".format(data_input_folder), sheet_name = 0)
df_test = pd.read_excel("{}\\genres_bi.xlsx".format(data_input_folder), sheet_name = 1)

X_train = df_train.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_train = df_train["class"]

X_test = df_test.drop(["Unnamed: 0", "id", "artist name", "track name", "class"], axis = 1)
y_test = df_test["class"]


##################################
### Load results from analysis ###
##################################

## ML base model results 
with open("{}\\base_proba.pkl".format(base_input_folder), "rb") as fp:
    base_proba = pickle.load(fp) 
with open("{}\\base_results.pkl".format(base_input_folder), "rb") as fp:
    base_results = pickle.load(fp) 
with open("{}\\models_dict_base.pkl".format(base_input_folder), "rb") as fp:
    models_dict_base = pickle.load(fp)
with open("{}\\models_dict_stack.pkl".format(base_input_folder), "rb") as fp:
    models_dict_stack = pickle.load(fp)  
with open("{}\\base_prc.pkl".format(base_input_folder), "rb") as fp:
    base_prc = pickle.load(fp)    
with open("{}\\base_roc.pkl".format(base_input_folder), "rb") as fp:
    base_roc = pickle.load(fp)    
with open("{}\\base_prc_thresh.pkl".format(base_input_folder), "rb") as fp:
    base_prc_thresh = pickle.load(fp) 
    
## Stacked model results
with open("{}\\best_stacks.pkl".format(stack_input_folder), "rb") as fp:
    best_stacks = pickle.load(fp)    
with open("{}\\stack_preds.pkl".format(stack_input_folder), "rb") as fp:
    stack_preds = pickle.load(fp)
with open("{}\\stack_proba.pkl".format(stack_input_folder), "rb") as fp:
    stack_proba = pickle.load(fp)
with open("{}\\stack_results.pkl".format(stack_input_folder), "rb") as fp:
    stack_results = pickle.load(fp)     
with open("{}\\stack_models.pkl".format(stack_input_folder), "rb") as fp:
    stack_models = pickle.load(fp)
with open("{}\\stack_params.pkl".format(stack_input_folder), "rb") as fp:
    stack_params = pickle.load(fp)
with open("{}\\stack_roc.pkl".format(stack_input_folder), "rb") as fp:
    rocauc = pickle.load(fp)
with open("{}\\stack_params.pkl".format(stack_input_folder), "rb") as fp:
    stack_params = pickle.load(fp)
with open("{}\\prc_thresh.pkl".format(stack_input_folder), "rb") as fp:
    stack_prc_thresh = pickle.load(fp)
with open("{}\\stack_prc.pkl".format(stack_input_folder), "rb") as fp:
    stack_prc = pickle.load(fp)    
with open("{}\\stack_roc.pkl".format(stack_input_folder), "rb") as fp:
    stack_roc = pickle.load(fp)
with open("{}\\meta_X_train.pkl".format(stack_input_folder), "rb") as fp:
    meta_X_train = pickle.load(fp)    
with open("{}\\meta_X_test.pkl".format(stack_input_folder), "rb") as fp:
    meta_X_test = pickle.load(fp)
    
    
## Multiclass results
with open("{}\\base_results_multi.pkl".format(multi_input_folder), "rb") as fp:
    base_results_multi = pickle.load(fp)     
with open("{}\\stack_results_multi.pkl".format(multi_input_folder), "rb") as fp:
    stack_results_multi = pickle.load(fp)     
with open("{}\\stack_roc_multi.pkl".format(multi_input_folder), "rb") as fp:
    stack_roc_multi = pickle.load(fp)
   
## Dimension reduction results
with open("{}\\base_results_reduc.pkl".format(reduction_input_folder), "rb") as fp:
    base_results_reduc = pickle.load(fp)
    
#############################
### Binary Classification ###
#############################

### Optimal Threshold on PRC curve for base GBT Model ###
base_thresholds = {}
for mod in models_dict_base.keys():
    numer = (2 * base_prc_thresh[mod][0] * base_prc_thresh[mod][1])
    denom = (base_prc_thresh[mod][0]  + base_prc_thresh[mod][1] )
    fscore = np.divide(numer, denom, out=np.zeros_like(denom), where=(denom!=0)) #denominator of F1-Score can be zero, hence replace nan with 0 in fscroe
    idx = np.argmax(fscore) #locate largest F1 Score
    base_thresholds[mod] = base_prc_thresh[mod][2][idx] #save threshild that produces largest F1 Score

stack_thresholds = {}
for mod in stack_models.keys():
    numer = (2 * stack_prc_thresh[mod][0] * stack_prc_thresh[mod][1])
    denom = (stack_prc_thresh[mod][0]  + stack_prc_thresh[mod][1] )
    fscore = np.divide(numer, denom, out=np.zeros_like(denom), where=(denom!=0)) #denominator of F1-Score can be zero, hence replace nan with 0 in fscroe
    idx = np.argmax(fscore) #locate largest F1 Score
    stack_thresholds[mod] = stack_prc_thresh[mod][2][idx] #save threshild that produces largest F1 Score
best_stacks.to_latex(index=False, formatters={"Model": str.upper})
"""
#######################
### Model averaging ###
#######################

#Use two models for averaging for each lead type
averaging = {}
models = ["gbt", "qda"] 
averaging = pd.DataFrame(columns=["alpha", "f1"])
counter = 0    
#init dicts
proba_thresh = {}
prc_thresh = {}

for mod in models:
    # Use validation set to avoid data leakage on test set
    X1, X2, y1, y2 = train_test_split(meta_X_train[mod], y_train, test_size = 0.2, random_state=123)  
    
    stack_models[mod][0].fit(X1, y1) #fit model to training data
    proba_thresh[mod] = stack_models[mod][0].predict_proba(X2)[:, 1] # predict probabilities on validation data
    

for a in np.linspace(0, 1, 101):
    
    avg_thresh = a*stack_thresholds[models[0]] + (1-a)*stack_thresholds[models[1]]
    avg_proba = a*proba_thresh[models[0]] + (1-a)*proba_thresh[models[1]]
    avg_preds = copy.deepcopy(avg_proba)
    
    avg_preds[avg_preds>=stack_thresholds[mod]] = 1 #make predictions based on thresholds
    avg_preds[avg_preds<stack_thresholds[mod]] = 0
    
    f1 = f1_score(y2, avg_preds)
    averaging.loc[counter] = [a, f1]
    counter +=1
averaging = averaging.sort_values("f1", ascending=True).reset_index(drop=True)     
"""

##################################################
######## ROC Curve for all stacked models ########
##################################################
plt.rcParams["figure.figsize"] = (15, 10)

colors = ["navy", "turquoise", "darkorange", "red", "green", "yellow", "purple", "brown"]
counter = 0

for key in models_dict_base.keys():
    plt.plot(rocauc[key][0], rocauc[key][1], linewidth = 2, color = colors[counter], label="{}, AUC={}".format(key, rocauc[key][3]))
    counter = counter+1

plt.plot([0, 1], [0, 1], color = 'black', linewidth = 2, linestyle = "dotted")
plt.ylabel('True Positive Rate', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.legend(loc=4, prop={'size': 18})
plt.title(None)
plt.savefig("{}\\all_roc.png".format(output_folder), dpi = 800, bbox_inches='tight')
plt.clf()


##############################################################
### Stack Feature Importance plot for GBT and Logit Stacks ###
##############################################################

n = 6 #number of features
#GBT Feature Importance
feature_names = list(X_train.columns)
for x in ["LDA_preds", "KNN_preds", "GBT_preds", "DT_preds"]: feature_names.append(x)
booster = stack_models["gbt"][0].get_booster()
gbt_importance = booster.get_score(importance_type="gain")

gbt_gain = []
gbt_feature = []
for key in gbt_importance.keys():
    gbt_gain.append(round(gbt_importance[key],2))
    gbt_feature.append(key)
    
gbt_imp = pd.DataFrame({"Information Gain": gbt_gain, "Feature": gbt_feature} )    
gbt_imp.sort_values("Information Gain", ascending = False, inplace = True)
gbt_imp.reset_index(drop = True, inplace=True)

best_features = [int(i) for i in gbt_imp["Feature"].values[0:n]]
features_select = [feature_names[i] for i in best_features] 


# Build trace
gbt_trace = go.Scatter(
        x = gbt_imp["Information Gain"].values[0:n][::-1], # Only first n features 
        y = features_select[::-1],
        mode="markers+text",
        marker=dict(
            sizemode = "diameter",
            sizeref = 1,
            size = 20,
            color = gbt_imp["Information Gain"].values[0:n][::-1],
            colorscale="Portland",
            showscale=False
        ),
        text = gbt_imp["Information Gain"].values[0:n][::-1]
    )

data = [gbt_trace]

layout= go.Layout(
    autosize= True,
    title= None,
    hovermode= 'closest',
     xaxis= dict(
         title= 'Information Gain',
         ticklen= 5,
         zeroline= False,
         gridwidth= 2,
     ),
    yaxis=dict(
        title= 'Feature',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
fig.update_layout(
    font=dict(size=25)
    )

fig.update_traces(textposition='top center')
pio.write_image(fig, "{}\\importance.png".format(output_folder),scale=3, width=1500, height=1000)


######## Hyperparemter tuning plot for GBT Model #########

#Need to run hyperparameter tuning again because study was not saved before
# Tune models for each lead-type
tuner = objective(X_train, y_train) #define objective 

study_gbt = optuna.create_study(direction="maximize",
                                      sampler=optuna.samplers.TPESampler(seed=42),
                                      pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
study_gbt.optimize(tuner.gbt, n_trials=100) # start optimization for current model

plt.rcParams["figure.figsize"] = (15, 10)
plt.rc('axes', titlesize=50) 
opt_plot = optuna.visualization.matplotlib.plot_optimization_history(study_gbt, target_name = "F1 Score")

plt.savefig("{}\\tuning.png".format(output_folder), dpi = 800, bbox_inches='tight')
plt.clf()


#########################################################
### Optimal Threshold on PRC curve for base GBT Model ###
#########################################################

mod = "gbt"
numer = (2 * base_prc_thresh[mod][0] * base_prc_thresh[mod][1])
denom = (base_prc_thresh[mod][0]  + base_prc_thresh[mod][1] )
fscore = np.divide(numer, denom, out=np.zeros_like(denom), where=(denom!=0)) #denominator of F1-Score can be zero, hence replace nan with 0 in fscroe
idx = np.argmax(fscore) #locate largest F1 Score

display = PrecisionRecallDisplay.from_estimator(
    models_dict_base[mod][0], X_test, y_test, name=mod, plot_chance_level=True, drop_intermediate= True
)
_ = display.ax_.set_title("2-class Precision-Recall curve")
_ = display.ax_.plot(base_prc[mod][1][idx], base_prc[mod][0][idx], color = "black", marker = "o")
_ = display.figure_.savefig("{}\\thresholds.png".format(output_folder), dpi = 800, bbox_inches='tight')



######################################
###### Explorative Data Analysis #####
######################################

sns.set(font_scale=1.65)
sns.pairplot(df_train, hue = "class", hue_order = [1, 0], palette ='muted', vars = ["danceability", "instrumentalness", "acousticness"], height = 3, aspect=1.5)
plt.savefig("{}\\eda.png".format(output_folder), dpi = 800, bbox_inches='tight',pad_inches = 0)


plt.rcParams["figure.figsize"] = (20, 10)
sns.set(style="darkgrid", font_scale=1.65)
fig, ax = plt.subplots(1, 3)
sns.kdeplot(data=df_train, x="danceability", hue="class", hue_order = [1, 0], fill=True,palette ='muted', common_norm=False, alpha=0.4, ax=ax[0])
sns.kdeplot(data=df_train, x="instrumentalness", hue="class", hue_order = [1, 0], fill=True,palette ='muted', common_norm=False, alpha=0.4, ax=ax[1])
sns.kdeplot(data=df_train, x="acousticness", hue="class", hue_order = [1, 0], fill=True,palette ='muted', common_norm=False, alpha=0.4, ax=ax[2])
plt.show()
plt.savefig("{}\\eda.png".format(output_folder), dpi = 800, bbox_inches='tight',pad_inches = 0)
plt.clf()

x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2


hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']
colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    ff.create_distplot(hist_data, group_labels, colors=colors,
                             bin_size=.2, show_rug=False),
    row = 1, col = 1
)

fig.add_trace(
    ff.create_distplot(hist_data, group_labels, colors=colors,
                             bin_size=.2, show_rug=False),
    row = 1, col = 2
)

fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
fig.show()


hist_data = [x1, x2, x3]

group_labels = ['Group 1', 'Group 2', 'Group 3']
colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data, group_labels, colors=colors,
                         bin_size=.2, show_rug=False)

# Add title
fig.update_layout(title_text='Hist and Curve Plot')
fig.show()
 
"""

colors = ["navy", "turquoise", "darkorange", "red", "green", "darkgreen", "yellow", "purple", "brown", "grey", "black"]
lw = 2

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 10], target_names):
    plt.scatter(
        X_r2[y_train == i, 0], X_r2[y_train == i, 1], alpha=0.8, color=color, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title("LDA of IRIS dataset")
"""