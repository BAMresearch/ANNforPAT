# -*- coding: utf-8 -*-
"""

@author: Sascha Liehr

load training data, validation data and test data

data available at:  https://doi.org/10.5281/zenodo.3677139

"""

import numpy as np
from scipy.io import loadmat


def load_pure_component_spectra_training_data():
    # load pure component spectra dataset (NNi):
    X = np.concatenate((loadmat('PureComponentSpectraDataset01.mat')['A_Spec_all'][:,4:],loadmat('PureComponentSpectraDataset02.mat')['A_Spec_all'][:,4:], loadmat('PureComponentSpectraDataset03.mat')['A_Spec_all'][:,4:]))
    Y = np.concatenate((loadmat('PureComponentSpectraDataset01.mat')['A_Spec_all'][:,:4],loadmat('PureComponentSpectraDataset02.mat')['A_Spec_all'][:,:4], loadmat('PureComponentSpectraDataset03.mat')['A_Spec_all'][:,:4]))
    return X, Y


def load_spectral_model_training_data():    
    # load spectral model dataset (NNii):
    X = np.concatenate((loadmat('SpectralModelDataSet01.mat')['A_Spec_all'][:,4:],loadmat('SpectralModelDataSet01.mat')['A_Spec_all'][:,4:], loadmat('SpectralModelDataSet01.mat')['A_Spec_all'][:,4:]))
    Y = np.concatenate((loadmat('SpectralModelDataSet01.mat')['A_Spec_all'][:,:4],loadmat('SpectralModelDataSet01.mat')['A_Spec_all'][:,:4], loadmat('SpectralModelDataSet01.mat')['A_Spec_all'][:,:4]))
    return X, Y


def load_validation_data():
    # import measured spectra for validation during training (with high-field NMR results [:,0:4]):
    X_val_meas = loadmat('ExperimentalLowFieldNmrSpectra_valid.mat')['A_Spec_all'][:,8:]        # low-filed NMR validation spectra
    Y_val_meas = loadmat('ExperimentalLowFieldNmrSpectra_valid.mat')['A_Spec_all'][:,:4]        # high-field NMR validation results
    return X_val_meas, Y_val_meas


def load_test_data():
    # import measured spectra for testing after training (with high-field NMR results [:,0:4] and low-field NMR results [:,4:8]):
    X_test_meas = loadmat('ExperimentalLowFieldNmrSpectra_test.mat')['A_Spec_all'][:,8:]            # low-filed NMR test spectra
    Y_test_meas = loadmat('ExperimentalLowFieldNmrSpectra_test.mat')['A_Spec_all'][:,:4]            # high-field NMR test results
    Y_test_meas_nfNMR_IHM = loadmat('ExperimentalLowFieldNmrSpectra_test.mat')['A_Spec_all'][:,4:8] # low-filed NMR test results (IHM)
    return X_test_meas, Y_test_meas, Y_test_meas_nfNMR_IHM 
