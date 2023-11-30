import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import numpy as np 
import tensorflow as tf



import os
import random
from glob import glob
import shutil

seed_value= 7
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

if(len(glob("output_data"))):
	shutil.rmtree("output_data")
os.mkdir("output_data")

if(len(glob("latent_data"))):
	shutil.rmtree("latent_data")
os.mkdir("latent_data")

AE = tf.keras.models.load_model("../../wt30MM/save_model/save_model_hic_latent_dim_3.keras")

data = np.loadtxt("expt_pij_d_mkbf22MM_5kb.mat")
np.fill_diagonal(data, 1.0)
np.fill_diagonal(data[1:], 1.0)
np.fill_diagonal(data[:, 1:], 1.0)

output = AE.predict(data)

encoder = tf.keras.models.Model(AE.input, AE.layers[4].output) 
latent_data = encoder.predict(data)
np.savetxt(f"latent_data/latent_data_hic_latent_dim_3_delta_mukbf.txt", latent_data, delimiter = "\t", fmt = "%0.3e")

np.savetxt("output_data/output_matrix_delta_mukbf.mat", output, fmt = "%0.3e")

