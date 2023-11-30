import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import os
import random
import pickle
from tqdm import tqdm 
from scipy.stats import pearsonr
import sys
import glob

seed_value = 1234
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

try:
	trees = int(sys.argv[1])
except:
	print("please provide the number of trees in the forest and how many run has been taken for traing")
	print("exiting-----------")
	exit(0)



if not os.path.exists("predicted_output"):
	os.mkdir("predicted_output")

if not os.path.exists("correlation"):
	os.mkdir("correlation")

if not os.path.exists("error"):
	os.mkdir("error")


#~~~~~~~~~~~load train data
print("LOADING TRAIN AND TEST DATA")
testing_data  = pickle.load(open(f"final_testing_data_delmatp30MM.pkl", "rb"))



##load the time
reqtime = np.genfromtxt(f'particular_time_run_151.txt')
reqtime = reqtime[1:]

##~~~~~~~~~~~~previously traing data has features and labels now extract the features
##train_X contains all the features of all particles means (runs*n_part, 928)
test_x = [i[0] for i in testing_data]
test_x = np.array(test_x)
print(test_x.shape)


##check if there any nan or infinity in the training and testing features
has_nan = np.isnan(test_x).any()
has_inf = np.isinf(test_x).any()
if has_nan:
    print("it finds nan in test data")
if has_inf:
    print("it finds inf in test data")

lag_time = []
final_rho = []
error_mae = []
error_mse = []
error_rsqrt = []




print("TIME LOOP STARTED")
##~~~~~~~~~~~~~~~~~~~~~~~Here the loop over time for traing and testing~~~~~~~~~~~~~~~~~~~~~~~~
pbar = tqdm(total = int(len(reqtime)), desc = "Progress")
for t in range(0, len(reqtime), 1):
# for t in range(0, 2, 1):
	test_y = [i[1][t] for i in testing_data]
	test_y = np.array(test_y)
	# print(train_y.shape, test_y.shape)

	##check if there any nan or infinity in the training and testing labels

	has_nan = np.isnan(test_y).any()
	has_inf = np.isinf(test_y).any()
	if has_nan:
		print("it finds nan in test labels")
	if has_inf:
		print("it finds inf in test labels")
	
	model = pickle.load(open(f"../wt30MM/save_model/rf_regression_model_time_{reqtime[t]}_trees_{trees}.pkl", "rb"))

	y_out_test  = model.predict(test_x)
	pickle.dump(y_out_test, open(f'predicted_output/rf_regression_delmatp30MM_predicted_output_time_{reqtime[t]}_trees_{trees}.pkl', 'wb'))

	mae = mean_absolute_error(test_y, y_out_test) 
	mse = mean_squared_error(test_y, y_out_test)
	rsqrt = r2_score(test_y, y_out_test)
	error_mae.append(mae)
	error_mse.append(mse)
	error_rsqrt.append(rsqrt)

	pear = np.corrcoef(test_y, y_out_test)[0,1]
	corr = pearsonr(test_y, y_out_test)[0]
	final_rho.append(pear)
	print(pear, corr)

	lag_time.append(reqtime[t])
	pbar.update()

final_rho = np.array(final_rho)
lag_time = np.array(lag_time)

error_mae = np.array(error_mae)
error_mse = np.array(error_mse)
error_rsqrt = np.array(error_rsqrt)

np.savetxt(f"correlation/predicted_delmatp30MM_rho_trees_{trees}_rf.txt", np.array([lag_time, final_rho]).T,
	delimiter = "\t", fmt = "%0.3e")
np.savetxt(f"error/different_error_delmatp30MM_trees_{trees}_rf.txt", np.array([error_mae, error_mse, error_rsqrt]).T,
	delimiter = "\t", fmt = "%0.3e")
pbar.close()
plt.show()
     
