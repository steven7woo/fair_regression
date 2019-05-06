import run_exp as run_exp
import numpy as np
import solvers as solvers


# Global Variables
Theta = np.linspace(0, 1.0, 41)  # the set of thresholds used in the algorithm
alpha = (Theta[1] - Theta[0])/2
DATA_SPLIT_SEED = 4
n = 2000 # subsample size; not used here.



# benchmark results that used fair classification
adult_FC_tree = pickle.load(open('adult_grid_tree.pkl', 'rb'))
adult_FC_lin = pickle.load(open('adult_grid_lin.pkl', 'rb'))

# here is an example on how to retrieve info
print('Learner:',  adult_FC_tree['learner'])
for lamb in adult_FC_tree['train_eval'].keys():
    print('Train Evaluation for lambda = %3f: average loss = %3f, DP_disp = %3f'% (lamb, adult_FC_tree['train_eval'][lamb]['average_loss'], adult_FC_tree['train_eval'][lamb]['DP_disp']))


# Sample instantiation of running the fair regeression algorithm
eps_list = [0.04, 0.06, 0.08]

dataset = 'adult'  # name of the data set
constraint = "DP"  # name of the constraint; so far limited to demographic parity (or statistical parity)
loss = "logistic"  # name of the loss function
learner = solvers.XGB_Classifier_Learner(Theta) # Specify a supervised learning oracle oracle 

info = str('Dataset: '+dataset + '; loss: ' + loss + '; eps list: '+str(eps_list)) + '; Solver: '+learner.name
print('Starting experiment. ' + info)

# Run the fair learning algorithm the supervised learning oracle
result = run_exp.fair_train_test(dataset, n, eps_list, learner,
                          constraint=constraint, loss=loss,
                          random_seed=DATA_SPLIT_SEED)

run_exp.read_result_list([result])  # A simple print out for the experiment




