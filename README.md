# Hi-C_ML_structure-dynamics

# Code Requirements

Ensure you have the following Python packages installed to run the code:

1. [numpy](https://numpy.org/)
2. [tensorflow](https://www.tensorflow.org/)
3. [scikit-learn](https://scikit-learn.org/stable/)

## Structure Folder

Inside the **structure** folder, you'll find three subfolders containing Autoencoder scripts:
### Autoencoder Scripts

1. **wt30MM/autoencoder_Hic.py**
   - Python script to train the autoencoder for wild-type Hi-C matrix.

2. **delmatp30MM/predicted_from_wt30MM_training/auto_delta_matp.py**
   - Python script to recreate the Hi-C matrix for the $\Delta$ MatP mutant using a trained model on WT30MM data.

3. **delmukbf22MM/predicted_from_wt30MM_training/auto_delta_mukbf.py**
   - Python script to recreate the Hi-C matrix for the $\Delta$ MukBEF mutant using a trained model on WT30MM data.

## Dynamics Folder
Within the **dynamics** folder, you'll find three subfolders containing Random Forest Regression scripts:
### Random Forest regression Scripts

1. **wt30mm/hic_random_forest.py**
   - Python scripts for training and predicting the dynamics of wild-type chromosomes.

2. **delmatp30MM/hic_random_forest_predict_delmatp.py**
   - Python scripts to predict the dynamics for the $\Delta$ MatP mutant using a trained model on WT30MM data.

3. **delmukbf22MM/hic_random_forest_predict_delmukbef.py**
   - Python scripts to predict the dynamics for the $\Delta$ MukBEF mutant using a trained model on WT30MM data.
