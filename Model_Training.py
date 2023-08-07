#%% -----------------------------Imports-----------------------------------
import pyxdf
import PyQt5
import mne
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
import pathlib
import mne
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#from joblib import dump,load
import pickle
# For interactive plots
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
#from subfunctions import *

from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
from matplotlib import pyplot as plt

# MNE functions
from mne import Epochs,find_events
from mne.decoding import Vectorizer


# Scikit-learn and Pyriemann ML functionalities
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from pyriemann.estimation import ERPCovariances, XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from sklearn.model_selection import train_test_split
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class BandwidthFilter: # A class for bandwidth filter
    def __init__(self, l_freq, h_freq, method):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        
class model: # A class holding relevent information regarding thr model
    def __init__(self, clsf, input_shape, ch_names,bandwidthfilter,conditions_dict):
        self.clsf = clsf
        self.input_shape = input_shape
        self.ch_names = ch_names
        self.bandwidthfilter = bandwidthfilter
        self.conditions_dict = conditions_dict


#%% -----------------------------Load Data-----------------------------------

binary_classification = False # For non-Target vs Target only

processed_file_name = 'Omri_2507'
#processed_file_name = 'or_1304'
current_path = pathlib.Path().absolute()  
data_fname = current_path /'Data'/'Processed Data'/ (processed_file_name + '_Processed.fif')
epochs = mne.read_epochs(data_fname)

#read the filter file
picklefile = open(current_path /'Data'/'Processed Data'/ (processed_file_name+'_Filter'), 'rb')
#unpickle the dataframe
bandwidth_filter = pickle.load(picklefile)
#close file
picklefile.close()


if binary_classification == True:
    if 'Standard Trial' in epochs.event_id: # Cheak if the data has been collected by the old paradigm (standard vs target)
        epochs =  epochs['Target Trial','Standard Trial'] 
    else: 
        epochs =  epochs['Target Trial','Non-Target Trial']

#bandwidth_filter = load(current_path /'Data'/'Processed Data'/ (processed_file_name+'_Filter'))
# %%
X = epochs.get_data()  # EEG signals: n_epochs, n_channels, n_times
X = epochs.get_data() * 1e6 # mV
# Remove baslineperiod
baseline_duration = epochs.tmin # get the baseline period
onset_sample = int(np.absolute(baseline_duration) * 125)
X = X[:,:,onset_sample::]

y = epochs.events[:, 2]   # get lables for classification 

conditions_dict =  {v: k for k, v in epochs.event_id.items()} # Map labels to strings 



# %%  --------------------------- Models creations and cross validation ------------------------
if binary_classification == True:
    class_weights = dict.fromkeys(conditions_dict, 0.5)
else:
    class_weights = {epochs.event_id['Distractor Trial']:1,epochs.event_id['Target Trial']:4,epochs.event_id['Non-Target Trial']:4}

clfs = OrderedDict() # holds all the models as a dict

clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
clfs['Xdawn + RegLDA'] = make_pipeline(Xdawn(2, classes=[1]), Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())

# define cross validation
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42) # 10-folds cross-validation

# run cross validation for each pipeline
auc = []
methods = []
for m in clfs:
    res = cross_val_score(clfs[m], X, y==epochs.event_id['Target Trial'], scoring='roc_auc', cv=cv, n_jobs=-1) # get the results 
    auc.extend(res)
    methods.extend([m]*len(res))

results = pd.DataFrame(data=auc, columns=['AUC']) # make a data frame out of the results
results['Method'] = methods

# Get the most accurate classifier 
best_clfs_name =  results.groupby('Method', as_index=False)['AUC'].mean().max()['Method']

# Plot AUC
plt.figure(figsize=[8,4])
sns.barplot(data=results, x='AUC', y='Method')
plt.title(f'Most accurate classifer: {best_clfs_name}')
plt.xlim(0.2, 0.85)
sns.despine()
plt.tight_layout()
plt.show()

#%% Fit the models and show confusion matrix

train_ratio = 0.75
test_ratio = 0.25


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=1 - train_ratio) # splits for train and test
fig, axs = plt.subplots(ncols= clfs.__len__() , figsize=(20, 10), sharey='row')

#Train and test the model
for i,m in enumerate(clfs):
    clfs[m].fit(X_train, Y_train) # fit the model
    preds_rg  = clfs[m].predict(X_test) # get predictions
    # Printing the results
    acc2         = np.mean(preds_rg == Y_test)
    print(f"{m} Classification accuracy: %f " % (acc2))
    
    disp = ConfusionMatrixDisplay(confusion_matrix(Y_test,preds_rg),display_labels = sorted(epochs.event_id)) # disply confution matrix
    disp.plot(ax= axs[i],xticks_rotation=45)
    disp.im_.colorbar.remove()
    
    axs[i].set_title(m)

plt.show()

#%%------------------------------- Save the selected model------------------------------

pipe = clfs[best_clfs_name] 

pipe.fit(X,y)
input_shape = X.shape
ch_names = epochs.ch_names


m = model(pipe,input_shape,ch_names,bandwidth_filter,conditions_dict)


fname = processed_file_name+'_model'
path_fname = current_path /'Models'/ fname


#create a pickle file
picklefile = open(path_fname, 'wb')
#pickle the dictionary and write it to file
pickle.dump(m, picklefile)
#close the file
picklefile.close()

#%% Load the selected model

#read the pickle file
picklefile = open(path_fname, 'rb')
#unpickle the dataframe
loaded = pickle.load(picklefile)
#close file
picklefile.close()

# %% Models graveyard Old Models


#clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(class_weight=class_weights))
#clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
#clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())
#clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression(class_weight=class_weights))

