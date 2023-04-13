import pickle
import pathlib
from subfunctions import model,BandwidthFilter

# Read Model
current_path = pathlib.Path().absolute() 
model_fname = 'Omri_Recording_001'

fname = model_fname+'_model'
path_fname = current_path /'Models'/ fname


#read the pickle file
picklefile = open(path_fname, 'rb')
#unpickle the dataframe
loaded = pickle.load(picklefile)
#close file
picklefile.close()


clsf = loaded.clsf[2]

# Working from lab pc