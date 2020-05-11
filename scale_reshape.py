# -*- coding: utf-8 -*-
"""

@author: Sascha Liehr
"""

import numpy as np


def scale_add_channels(X,X_val_meas,X_test_meas,Y,Y_val_meas,Y_test_meas,Y_test_meas_nfNMR_IHM,X_scaling_factor,label_factor):
    ### scale all spectra:
    X = np.float32(X/X_scaling_factor)
    X_val_meas = np.float32(X_val_meas/X_scaling_factor)
    X_test_meas = np.float32(X_test_meas/X_scaling_factor)
    
    ### scale labels/ground truth:
    Y = np.float32(Y/label_factor)
    Y_val_meas = np.float32(Y_val_meas[:,:4]/label_factor) 
    Y_test_meas = np.float32(Y_test_meas[:,:4]/label_factor) 
    Y_test_meas_nfNMR_IHM = np.float32(Y_test_meas_nfNMR_IHM[:,:4]/label_factor)
    
    ### add channel dimension:
    X = X.reshape((X.shape[0],X.shape[1],1))
    X_val_meas = X_val_meas.reshape((X_val_meas.shape[0],X_val_meas.shape[1],1))
    X_test_meas = X_test_meas.reshape((X_test_meas.shape[0],X_test_meas.shape[1],1))
    
    return X,X_val_meas,X_test_meas,Y,Y_val_meas,Y_test_meas,Y_test_meas_nfNMR_IHM
    
