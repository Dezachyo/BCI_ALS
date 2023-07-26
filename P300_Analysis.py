#%% -----------------------------Imports ---------------------------
import pyxdf
import PyQt5
import mne
import numpy as np
import pathlib
import mne
import pickle
import matplotlib
import matplotlib.pyplot as plt
# For interactive plots
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from subfunctions import *
# %%
#processed_file_name = 'Or_1304'
processed_file_name = 'Omri_2507'

current_path = pathlib.Path().absolute()  
data_fname = current_path /'Data'/'Processed Data'/ (processed_file_name + '_Processed.fif')
epochs = mne.read_epochs(data_fname)
epochs

# %% -------------------------Create Evokeds dict for plotting

target_evoked = epochs['Target Trial'].average()
nontarget_evoked = epochs['Non-Target Trial'].average()
distractor_evoked = epochs['Distractor Trial'].average()

evokeds = dict(nontarget=nontarget_evoked, target=target_evoked,distractor =distractor_evoked)


# %% --------------------- Verify cleaned epochs (optional)
fig, ax = plt.subplots(3,2)

epochs['Target Trial'].plot_image(picks='eeg', combine='mean',axes=ax[:,0],title="Target")
epochs['Non-Target Trial'].plot_image(picks='eeg', combine='mean',axes=ax[:,1],title="Standard")

plt.show()


# %% --------------------------- Evoked in diffrent plots
fig, ax = plt.subplots(2)

P300_window = [0.25,0.4]
par_picks = ['CP1','CP2'] # to plot only parietal


target_evoked.plot(gfp=True,
    highlight=P300_window,axes=ax[0],titles='odd')

nontarget_evoked.plot(gfp=True,
    highlight=P300_window,axes=ax[1],titles='standard')

plt.show()


# %% ----------------------------- Compare Evoked (P300 Effect) --------------------------

mne.viz.plot_compare_evokeds(evokeds ,combine='mean',show_sensors= True,vlines= [0.3] )

# %% -------------------------------- ERP Topoplots ----------------------------------------

mne.viz.plot_compare_evokeds(evokeds,axes='topo',vlines=[0,0.3])

# %% 

Difference_wave = mne.combine_evoked([target_evoked,nontarget_evoked],weights=[1,-1])
mne.viz.plot_compare_evokeds(dict(nontaget=nontarget_evoked, target=target_evoked, difference=Difference_wave),axes='topo',vlines=[0,0.3])

mne.viz.plot_compare_evokeds(evokeds,combine='mean',show_sensors= True,vlines= [0.3] )

# %% ------------------------ Joint plots
par_picks = ['CP1','CP2'] # to plot only parietal

odd_evoked =  epochs['Target Trial'].average()
stand_evoked =  epochs['Standard Trial'].average()


odd_evoked.plot_joint(picks='eeg',times=[0,0.1,0.2,0.3])
