# -*- coding: utf-8 -*-
"""
@author: Sascha Liehr

Code accompanying the paper "Artificial neural networks for quantitative online NMR spectroscopy", Anal Bioanal Chem (2020)
Authors: Simon Kern, Sascha Liehr, Lukas Wander, Martin Bornemann-Pfeiffer, Simon Müller, Michael Maiwald, Stefan Kowarik

publication link:   https://doi.org/10.1007/s00216-020-02687-5
data available at:  https://doi.org/10.5281/zenodo.3677139

"""

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import mean_squared_error
import plot
import load_data
import scale_reshape
import model_def


### training parameter and model architecture:
epochs = 400            # number of training epochs
learning_rate = 0.0001  # learning rate
batch_size = 1024       # mini batch size during training
strides = 9             # CNN strides
kernel_size = 9         # size of 1D convolutional kernels
filters = 4             # number of convolutional kernels
noise = 0.04            # noise factor for additive white Gaussian noise


# save models here:
model_path_name = r"specify_path_and_file_name"


### load data:
#X,Y = load_data.load_pure_component_spectra_training_data()  # load pure component spectra training dataset (NNi) 
X,Y = load_data.load_spectral_model_training_data()           # OR load spectral model training dataset (NNii):
X_val_meas, Y_val_meas = load_data.load_validation_data()     # load validation data
X_test_meas, Y_test_meas, Y_test_meas_nfNMR_IHM = load_data.load_test_data()  # load test data

label_factor = np.max(Y)      # compute label scaling factor
X_scaling_factor = np.max(X)  # compute input scaling factor


### scale and reshape spectra, labels and ground truth / add channel dimension:
X,X_val_meas,X_test_meas,Y,Y_val_meas,Y_test_meas,Y_test_meas_nfNMR_IHM = scale_reshape.scale_add_channels(X,X_val_meas,X_test_meas,Y,Y_val_meas,Y_test_meas,Y_test_meas_nfNMR_IHM,X_scaling_factor,label_factor)


### add noise to training data:
X = X + np.random.normal(0,noise,(np.shape(X)))


### build and compile model:
model = model_def.CNN_model(filters,kernel_size,X.shape[1:],strides)
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay=0, amsgrad=False)
model.compile(loss='mse', optimizer=optimizer)
model.summary()


### define callbacks:     
class callback_validation_meas_data(Callback):
     def __init__(self,epochs,X_val_meas,Y_val_meas,label_factor,model):
        self.mse_val_meas = np.zeros((epochs))
        self.prediction_result = np.zeros((epochs-1,Y_val_meas.shape[0],4))
        self.X_val_meas = X_val_meas
        self.Y_val_meas = Y_val_meas
        self.label_factor = label_factor
        self.model = model
        
     def on_epoch_end(self,epoch,logs={}):
        self.mse_val_meas[epoch] = mean_squared_error(self.Y_val_meas*self.label_factor, model.predict(self.X_val_meas)*self.label_factor)
        if epoch>0:
            self.prediction_result[epoch-1,:,:] = model.predict(X_val_meas)*self.label_factor
            if np.min(self.mse_val_meas[:epoch])>self.mse_val_meas[epoch]: 
                print ("mse_val_meas improved from ",np.round(np.min(self.mse_val_meas[:epoch]),5)," to ",np.round(self.mse_val_meas[epoch],5))
            else:
                print ("mse_val_meas = "+str(np.round(self.mse_val_meas[epoch],5)))
        else:
            print ("mse_val_meas ="+str(np.round(self.mse_val_meas[epoch],5)))
            
predictions = callback_validation_meas_data(epochs,X_val_meas,Y_val_meas,label_factor,model)
checkpointer = ModelCheckpoint(filepath=model_path_name+'.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
lrate = LearningRateScheduler(model_def.step_decay)
callbacks_list = [checkpointer, predictions, lrate]


### fit model:
hist = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=2, callbacks=callbacks_list, shuffle=True)


loss = hist.history['loss']
val_loss = hist.history['val_loss']
mse_val_meas = predictions.mse_val_meas  # mean squared error of validation data (used during training)


### compute mean squared error of validation data for each reactant during training:
mse_val_meas_all = np.zeros((predictions.prediction_result.shape[0],predictions.prediction_result.shape[2]))
for j in range(predictions.prediction_result.shape[0]):
    for k in range(predictions.prediction_result.shape[2]):
        mse_val_meas_all[j,k] =  mean_squared_error(Y_val_meas[:,k]*label_factor, predictions.prediction_result[j,:,k])


### predict concentration results using CNN model:
CNN_prediction = model.predict(X_test_meas)*label_factor


### plot predictions and training progress
plot.plot(Y_test_meas,label_factor,Y_test_meas_nfNMR_IHM,CNN_prediction,epochs,mse_val_meas_all,mse_val_meas,loss,val_loss)
