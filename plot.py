# -*- coding: utf-8 -*-
"""
@author: Sascha Liehr

### plot low-field IHM result, low-field ANN result and high-field IHM result:
"""
import numpy as np
from matplotlib import pyplot as plt

def plot(Y_test_meas,label_factor,Y_test_meas_nfNMR_IHM,CNN_prediction,epochs,mse_val_meas_all,mse_val_meas,loss,val_loss):
    ### plot low-field IHM result, low-field ANN result and high-field IHM result:
    font_size = 13
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.title('$\ito$-FNB:',loc='left',fontsize=font_size)
    plt.plot(Y_test_meas[:,0]*label_factor,label='measured (HF NMR)')
    plt.plot(Y_test_meas_nfNMR_IHM[:,0]*label_factor,label='measured (IHM)')
    plt.plot(CNN_prediction[:,0],label='CNN prediction')
    plt.xlim((0,Y_test_meas.shape[0]))
    plt.grid(linestyle='--')
    plt.ylabel('Component areas $\it{A}$',fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.subplot(222)
    plt.title('Li-MNDPA:',loc='left',fontsize=font_size)
    plt.plot(Y_test_meas[:,1]*label_factor,label='measured (HF NMR)')
    plt.plot(Y_test_meas_nfNMR_IHM[:,1]*label_factor,label='measured (IHM)')
    plt.plot(CNN_prediction[:,1],label='CNN prediction')
    plt.xlim((0,Y_test_meas.shape[0]))
    plt.grid(linestyle='--')
    plt.legend(fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.subplot(223)
    plt.title('Li-Toluidine:',loc='left',fontsize=font_size)
    plt.plot(Y_test_meas[:,2]*label_factor,label='measured (HF NMR)')
    plt.plot(Y_test_meas_nfNMR_IHM[:,2]*label_factor,label='measured (IHM)')
    plt.plot(CNN_prediction[:,2],label='CNN prediction')
    plt.xlim((0,Y_test_meas.shape[0]))
    plt.grid(linestyle='--')
    plt.legend(fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.xlabel('No. of spectra',fontsize=font_size)
    plt.ylabel('Component areas $\it{A}$',fontsize=font_size)
    plt.subplot(224)
    plt.title('$\itp$-Toluidine:',loc='left',fontsize=font_size)
    plt.plot(Y_test_meas[:,3]*label_factor,label='measured (HF NMR)')
    plt.plot(Y_test_meas_nfNMR_IHM[:,3]*label_factor,label='measured (IHM)')
    plt.plot(CNN_prediction[:,3],label='CNN prediction')
    plt.xlim((0,Y_test_meas.shape[0]))
    plt.grid(linestyle='--')
    plt.xlabel('No. of spectra',fontsize=font_size)
    plt.legend(fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.tight_layout()
    
    # plot training progress (mean squared errors and loss/validation loss during training)
    plt.figure(figsize=(10,8))
    plt.subplot(211)
    plt.title(r'CNN training')
    plt.plot(np.arange(1,epochs),mse_val_meas_all[:,0],color='g', label='$\ito$-FNB') # mse Einzelstoffe
    plt.plot(np.arange(1,epochs),mse_val_meas_all[:,1],color='b', label='Li-MNDPA')
    plt.plot(np.arange(1,epochs),mse_val_meas_all[:,2],color='pink',label='Li-Toluidine')
    plt.plot(np.arange(1,epochs),mse_val_meas_all[:,3],color='red',label='$\itp$-Toluidine')
    plt.plot(np.arange(1,epochs),mse_val_meas[1:], linewidth=3, color='black', label='all reactants')
    plt.legend(loc='best',fontsize=font_size)
    plt.grid(linestyle='--')
    plt.xlabel('Epoch no.',fontsize=font_size)
    plt.ylabel(r'MSE($\it{A}\rm_{NN\it_{i}\rm{,val}}$,$\it{A}\rm_{hf,val}$)',fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.subplot(212)
    plt.plot(loss,label=r'loss NN$\it_{i}$')
    plt.plot(val_loss,label=r'val. loss NN$\it_{i}$')
    plt.legend(loc='best',fontsize=font_size)
    plt.grid(linestyle='--')
    plt.xlabel('Epoch no.',fontsize=font_size)
    plt.ylabel('MSE loss / MSE val. loss',fontsize=font_size)
    plt.tick_params(axis='both', labelsize=font_size)
    plt.tight_layout()
    
    return 1
