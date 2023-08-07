#%% -----------------------------Imports ---------------------------

# Some standard pythonic imports
import warnings
warnings.filterwarnings('ignore')
import os,numpy as np,pandas as pd
from collections import OrderedDict
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import pathlib
from os import listdir
from os.path import isfile, join
import pyxdf
import PyQt5
import pickle
# MNE functions
import mne
from mne import Epochs,find_events
# Custom function  
from subfunctions import *
# For interactive plots
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
# %% -------------------- Read Data -----------------------------------

fname = 'Omri_2507.xdf' # file name
current_path = pathlib.Path().absolute()  

## Recording to XDF
current_path = pathlib.Path().absolute()  
data_fname = current_path /'Data'/fname
# Add annotation from events
raw,events = read_xdf(fname = data_fname)
raw = add_annot(raw, events)
# %% --------------------- Bandpass filtering ---------------------------
[LowPass, HighPass] = 1 , 40

LowPass = float(LowPass)
HighPass = float(HighPass)
filter_method = 'iir'

selected_filter = BandwidthFilter(LowPass, HighPass, filter_method)
Raw_Filtered = raw.filter(LowPass, HighPass, method=filter_method)
Raw_Filtered.notch_filter([25])
print (f'Bandpass filter {filter_method} [{int(LowPass)}-{int(HighPass)} Hz]')

# %% --------------------- Remove bad channels-------------------------------

Raw_Filtered.plot()
# %% --------------------- Epoching-------------------------------
Raw_Filtered.drop_channels(Raw_Filtered.info['bads']) # drop bad channels
Raw_Filtered.set_eeg_reference(ref_channels='average') # rereferance to average referance
events_from_annot, event_dict = mne.events_from_annotations(Raw_Filtered) # read markers
if 'Break' in event_dict:
    event_dict.pop('Break')
epochs = mne.Epochs(Raw_Filtered, events_from_annot, tmin=-0.2, tmax=0.5, event_id=event_dict,detrend=1,baseline= (-0.2,0), preload = True) # make epochs around the events and baseline correct
# %% --------------------- Drop Noisy Epochs-------------------------------
epochs.plot()
# %% --------------------- Verify cleaned epochs (optional)
fig, ax = plt.subplots(3,2)

epochs['Target Trial'].plot_image(picks='eeg', combine='mean',axes=ax[:,0],title="Target")
epochs['Non-Target Trial'].plot_image(picks='eeg', combine='mean',axes=ax[:,1],title="Non-Target Trial")
# %% ---------------------- Save preprocced file as .fif ----------
fif_export_path = current_path/'Data'/'Processed Data'/f'{fname[:-4]}_Processed.fif'
epochs.save(fif_export_path,overwrite=True)
bwfilter_export_path = current_path/'Data'/'Processed Data'/f'{fname[:-4]}_Filter'

with open(bwfilter_export_path, "wb") as file:
    pickle.dump(selected_filter, file)

# %%
