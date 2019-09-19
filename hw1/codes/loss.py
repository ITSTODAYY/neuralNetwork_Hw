from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''ELoss forward'''
        return np.sum(np.square(target-input))/(2*len(target))

    def backward(self, input, target):
        '''ELoss backward'''
        return input - target


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name
        self.saved_tensor = 0

    def forward(self, input, target):
        '''softmax forward'''
        #stepone:softmax
        #print(input.shape)
        output = np.exp(input)/(np.sum(np.exp(input),axis=1).reshape([input.shape[0],1]))
        #print(output)
        self.saved_tensor = np.exp(input)/(np.sum(np.exp(input),axis=1).reshape([input.shape[0],1]))
        #steptwo:cross
        output = - np.sum(target*np.log(output))
        return output

    def backward(self, input, target):
        '''SoftmaxBackward'''
        return self.saved_tensor - target

        
