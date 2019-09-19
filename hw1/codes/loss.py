from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        '''ELoss forward'''
        m = len(target)
        return np.sum(np.square(target-input))/(2*m)

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
        output = np.exp(input)/(np.squeeze(np.sum(np.exp(input))))
        self.saved_tensor = np.exp(input)/(np.squeeze(np.sum(np.exp(input))))
        #steptwo:cross
        output = - np.sum(target*np.log(input))
        return output

    def backward(self, input, target):
        '''SoftmaxBackward'''
        return self.saved_tensor - target

        
