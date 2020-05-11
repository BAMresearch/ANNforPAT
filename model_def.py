# -*- coding: utf-8 -*-
"""

@author: Sascha Liehr

define model and learning rate step decay

"""
from tensorflow.keras.layers import Dense, Input, Flatten, LocallyConnected1D
from tensorflow.keras.models import Model


def CNN_model(filters,kernel_size,input_dim,strides):    
    input = Input(shape=input_dim)
    x = LocallyConnected1D(filters=filters, kernel_size=kernel_size, strides=strides, activation='elu', padding='valid')(input)
    x = Flatten()(x)   
    outputs = Dense(4, activation='relu')(x)
    model = Model(inputs=[input], outputs=outputs)    
    return model


# learning rate schedule:
def step_decay(epoch):
    lrate_vector = [0.0001,0.00003,0.00001]
    step_epochs = [170,320]    
    if epoch<step_epochs[0]:
        lrate = lrate_vector[0]
    elif epoch<step_epochs[1]:
        lrate = lrate_vector[1]
    else:
        lrate = lrate_vector[2]
    return lrate
