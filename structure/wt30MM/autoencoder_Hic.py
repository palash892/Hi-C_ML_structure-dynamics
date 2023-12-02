import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import numpy as np
import tensorflow as tf


import os
import random
from glob import glob
import shutil



if(len(glob("save_model"))):
       shutil.rmtree("save_model")
os.mkdir("save_model")


if(len(glob("training_loss"))):
       shutil.rmtree("training_loss")
os.mkdir("training_loss")



if(len(glob("output_data"))):
       shutil.rmtree("output_data")
os.mkdir("output_data")


if(len(glob("latent_data"))):
       shutil.rmtree("latent_data")
os.mkdir("latent_data")

if(len(glob("model_loss"))):
       shutil.rmtree("model_loss")
os.mkdir("model_loss")

seed_value= 7
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)




data = np.loadtxt("expt_pij_wt30MM_5kb.mat")
np.fill_diagonal(data, 1.0)
np.fill_diagonal(data[1:], 1.0)
np.fill_diagonal(data[:, 1:], 1.0)

# mean = data.mean(axis = 0)
# data -= mean 
# std = data.std(axis = 0)
# data /= std
print(data.shape)


dimension = []
fraction = []
rec_loss = []

for f in range(1, 11, 1):

       input_dim = 928
       hidden_dim1 = 500
       hidden_dim2 = 200
       hidden_dim3 = 100
       latent_dim = int(f)
       output_dim = 928

       input_layer = tf.keras.Input(shape=(input_dim,), name = "INPUT")

       hidden_layer1 = tf.keras.layers.Dense(hidden_dim1, activation='leaky_relu') (input_layer)

       hidden_layer2 = tf.keras.layers.Dense(hidden_dim2, activation='leaky_relu') (hidden_layer1)

       hidden_layer3 = tf.keras.layers.Dense(hidden_dim3, activation='leaky_relu') (hidden_layer2)

       encoded = tf.keras.layers.Dense(latent_dim, activation='leaky_relu', name = "BOTTLE") (hidden_layer3)

       decoded1 = tf.keras.layers.Dense(hidden_dim3, activation='leaky_relu')(encoded)

       decoded2 = tf.keras.layers.Dense(hidden_dim2, activation='leaky_relu')(decoded1)

       decoded3 = tf.keras.layers.Dense(hidden_dim1, activation='leaky_relu')(decoded2)

       output_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid', name = "OUTPUT")(decoded3)

       AE = tf.keras.models.Model(input_layer, output_layer)


       # optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
       AE.compile(optimizer="adam", loss='binary_crossentropy')

       batch_size = 30
       epochs = 100
       X_train = data
       history = AE.fit(X_train, X_train, epochs=epochs, shuffle=True, batch_size=batch_size, 
              validation_data=(X_train, X_train))

       ##saving the model parameter
       AE.save(f"save_model/save_model_hic_latent_dim_{latent_dim}.h5")
       AE.save(f"save_model/save_model_hic_latent_dim_{latent_dim}.keras")

       ##calculating the training loss
       train_loss = history.history['loss']
       train_loss = np.array(train_loss)
       num_epochs = [int(value) for value in np.linspace(1, epochs, epochs) if value.is_integer()]
       num_epochs = np.array(num_epochs)
       np.savetxt(f"training_loss/training_loss_hic_latent_dim_{latent_dim}.txt", np.array([num_epochs, train_loss]).T, delimiter = "\t", fmt = "%0.3e")


       encoded_data = AE.predict(X_train)
       print(encoded_data.shape)
       np.savetxt(f"output_data/matrix_{latent_dim}.mat", encoded_data, fmt = "%0.3e")

       reconstruction_loss= AE.evaluate(X_train, X_train)
       print("reconstruction_loss = ", reconstruction_loss)
       rec_loss.append(reconstruction_loss)

       # Calculate the Fraction of Variance Explained (FVE)
       total_variance = np.sum((X_train - np.mean(X_train, axis = 0))**2)
       reconstruction_error = np.sum((X_train - encoded_data)**2)
       fve = 1 - (reconstruction_error / total_variance)

       print("Fraction of Variance Explained (FVE):", fve, latent_dim)
       dimension.append(latent_dim)
       fraction.append(fve)

       ##data from latent dimension
       encoder = tf.keras.models.Model(AE.input, AE.layers[4].output) 
       latent_data = encoder.predict(X_train)
       np.savetxt(f"latent_data/latent_data_hic_latent_dim_{latent_dim}.txt", latent_data, delimiter = "\t", fmt = "%0.3e")
       # print(latent_data)

dimension = np.array(dimension)
fraction = np.array(fraction)
np.savetxt("model_loss/fraction_of_variance_hic_type.txt", np.array([dimension, fraction]).T, fmt = "%0.3e", delimiter = "\t")
np.savetxt("model_loss/reconstruction_loss.txt", np.array([dimension, rec_loss]).T, fmt = "%0.3e", delimiter = "\t")
