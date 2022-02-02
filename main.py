import tensorflow as tf
import numpy as np
import argparse
import IPython.display as display
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# import cProfile
import copy as cp
from sklearn.model_selection import KFold

from data_construct import * ## contains functions for constructing data
from IRM_methods import *    ## contains IRM games methods


# Create data for each environment

n_e = 2  # number of environments

p_color_list = [0.2, 0.1] # list of probabilities of switching the final label to obtain the color index
p_label_list = [0.25]*n_e # list of probabilities of switching pre-label
D = assemble_data_mnist() # initialize mnist digits data object

D.create_training_data(n_e, p_color_list, p_label_list) # creates the training environments

p_label_test = 0.25 # probability of switching pre-label in test environment
p_color_test = 0.9  # probability of switching the final label to obtain the color index in test environment

D.create_testing_data(p_color_test, p_label_test, n_e)  # sets up the testing environment
(num_examples_environment,length, width, height) = D.data_tuple_list[0][0].shape # attributes of the data
num_classes = len(np.unique(D.data_tuple_list[0][1])) # number of classes in the data


# we use same architecture across environments and store it in a list
model_list = []
for e in range(n_e):
    model_list.append(keras.Sequential([
            keras.layers.Flatten(input_shape=(length, width,height)),
            keras.layers.Dense(390, activation = 'elu'),
             keras.layers.Dropout(0.75),
            keras.layers.Dense(390, activation='elu'),
             keras.layers.Dropout(0.75),
            keras.layers.Dense(num_classes)
    ]))


num_epochs       = 25
batch_size       = 256
termination_acc  = 0.6
warm_start       = 100
learning_rate    = 2.5e-4


# initialize F-IRM model (we pass the hyper-parameters that we chose above)
F_game = fixed_irm_game_model(model_list, learning_rate, num_epochs, batch_size, termination_acc, warm_start)

# fit function runs the training on the data that we created
F_game.fit(D.data_tuple_list)

# evaluate function runs and evaluates train and test accuracy of the final model
F_game.evaluate(D.data_tuple_test)

# print train and test accuracy
print ("Training accuracy " + str(F_game.train_acc))
print ("Testing accuracy " + str(F_game.test_acc))

plt.xlabel("Training steps")
plt.ylabel("Training accuracy")
plt.plot(F_game.train_accuracy_results)

model_list = [] # we use same architecture across environments and store it in a list and the last element of the list
# corresponds to the architecture for the representation learner
for e in range(n_e+1):
    if(e<=n_e-1):
        model_list.append( keras.Sequential([
            keras.layers.Flatten(input_shape=(390,1)),
            keras.layers.Dense(390, activation = 'elu'),
            keras.layers.Dropout(0.75),
            keras.layers.Dense(390, activation='elu'),
            keras.layers.Dropout(0.75),
            keras.layers.Dense(num_classes)
        ]))
    if(e==n_e):
        model_list.append(keras.Sequential([
        keras.layers.Flatten(input_shape=(length, width,height)),
        keras.layers.Dense(390, activation = 'elu',kernel_regularizer=keras.regularizers.l2(0.00125)),
      ]))

num_epochs       = 25
batch_size       = 256
termination_acc  = 0.55
warm_start       = 100
learning_rate    = 2.5e-4


# initialize V-IRM model (we pass the hyper-parameters that we chose above)
V_game = variable_irm_game_model(model_list, learning_rate, num_epochs, batch_size, termination_acc, warm_start)

# fit function runs the training on the data that we created
V_game.fit(D.data_tuple_list)

# evaluate function runs and evaluates train and test accuracy of the final model
V_game.evaluate(D.data_tuple_test)

# print train and test accuracy
print (V_game.train_acc)
print (V_game.test_acc)


plt.xlabel("Training steps")
plt.ylabel("Training accuracy")
plt.plot(V_game.train_accuracy_results)