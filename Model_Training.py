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

class BandwidthFilter:
    def __init__(self, l_freq, h_freq, method):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.method = method
        
class model:
    def __init__(self, clsf, input_shape, ch_names,bandwidthfilter,conditions_dict):
        self.clsf = clsf
        self.input_shape = input_shape
        self.ch_names = ch_names
        self.bandwidthfilter = bandwidthfilter
        self.conditions_dict = conditions_dict


#%% -----------------------------Load-----------------------------------

binary_classification = False

processed_file_name = 'Shahar_3_Class'
#processed_file_name = 'or_1304'
current_path = pathlib.Path().absolute()  
data_fname = current_path /'Data'/'Processed Data'/ (processed_file_name + '_Processed.fif')
epochs = mne.read_epochs(data_fname)

#read the pickle file
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
X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
X = epochs.get_data() * 1e6 # mV
# Remove baslineperiod
baseline_duration = -0.2
onset_sample = int(np.absolute(baseline_duration) * 125)
X = X[:,:,onset_sample::]

y = epochs.events[:, 2]  
#y = y - (max(y)-1) # 0's and 1's instead of 1's and 2's
conditions_dict =  {v: k for k, v in epochs.event_id.items()}



# %%
if binary_classification == True:
    class_weights = dict.fromkeys(conditions_dict, 0.5)
else:
    class_weights = {epochs.event_id['Distractor Trial']:1,epochs.event_id['Target Trial']:4,epochs.event_id['Non-Target Trial']:4}

clfs = OrderedDict()
#clfs['Vect + LR'] = make_pipeline(Vectorizer(), StandardScaler(), LogisticRegression(class_weight=class_weights))
clfs['Vect + RegLDA'] = make_pipeline(Vectorizer(), LDA(shrinkage='auto', solver='eigen'))
clfs['Xdawn + RegLDA'] = make_pipeline(Xdawn(2, classes=[1]), Vectorizer(), LDA(shrinkage='auto', solver='eigen'))

#clfs['XdawnCov + TS'] = make_pipeline(XdawnCovariances(estimator='oas'), TangentSpace(), LogisticRegression(class_weight=class_weights))
clfs['XdawnCov + MDM'] = make_pipeline(XdawnCovariances(estimator='oas'), MDM())


#clfs['ERPCov + TS'] = make_pipeline(ERPCovariances(), TangentSpace(), LogisticRegression())
#clfs['ERPCov + MDM'] = make_pipeline(ERPCovariances(), MDM())


#times = epochs.times


# define cross validation
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=42)

# run cross validation for each pipeline
auc = []
methods = []
for m in clfs:
    res = cross_val_score(clfs[m], X, y==epochs.event_id['Target Trial'], scoring='roc_auc', cv=cv, n_jobs=-1)
    auc.extend(res)
    methods.extend([m]*len(res))

results = pd.DataFrame(data=auc, columns=['AUC'])
results['Method'] = methods

plt.figure(figsize=[8,4])
sns.barplot(data=results, x='AUC', y='Method')
plt.xlim(0.2, 0.85)
sns.despine()

#%%

train_ratio = 0.75
test_ratio = 0.25

# train is now 50% of the entire data set
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=1 - train_ratio)
#Y_train =Y_train - (np.max(Y_train)-1)
#Y_test =Y_test - (np.max(Y_train)-1)
fig, axs = plt.subplots(clfs.__len__())

#logisticregression
for i,m in enumerate(clfs):
    clfs[m].fit(X_train, Y_train)
    preds_rg  = clfs[m].predict(X_test)
    # Printing the results
    acc2         = np.mean(preds_rg == Y_test)
    print(f"{m} Classification accuracy: %f " % (acc2))
    
    
    #fig,axs = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(Y_test,preds_rg),display_labels = sorted(epochs.event_id)).plot(ax= axs[i])
    axs[i].set_title(m)
plt.show()

#%%------------------------------- Save the selected model------------------------------

pipe = clfs['Vect + RegLDA'] 

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

#%%

#read the pickle file
picklefile = open(path_fname, 'rb')
#unpickle the dataframe
loaded = pickle.load(picklefile)
#close file
picklefile.close()


# %%

fig,axs = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(Y_test.argmax(axis = -1),preds)).plot(ax= axs)
axs.set_title('Title')

plt.tight_layout()

clfs['Vect + LR']
# %%
