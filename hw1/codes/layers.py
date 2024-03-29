import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Relu(Layer):
    def __init__(self, name):
        super(Relu, self).__init__(name)

    def forward(self, input):
        '''Relu Activation'''
        tensor = {
            "input" : input
        }
        self._saved_for_backward(tensor)
        return (np.abs(input)+input)/2

    def backward(self, grad_output):
        '''Relu Backward'''
        return grad_output*(self._saved_tensor["input"]>0)


class Sigmoid(Layer):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name)

    def forward(self, input):
        '''Sigmoid Activation'''
        output = 1/(1+np.exp(-input))
        tensor = {
            "output" : output
        }
        self._saved_for_backward(tensor)
        return output

    def backward(self, grad_output):
        '''Sigmoid backward'''
        return grad_output*(self._saved_tensor["output"]*(1-self._saved_tensor["output"]))


class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        """linear forward"""
        output = np.dot(input,self.W)+self.b
        tensor = {
            "input" : input,
            "output" : output
        }
        self._saved_for_backward(tensor) 
        #print(output.shape)     
        return output

    def backward(self, grad_output):
        '''Linear backward'''
        self.grad_W = np.dot(self._saved_tensor["input"].T,grad_output)
        #print(self.grad_W.shape)
        #print(grad_output.shape)
        #print(self.W.shape)
        #print(self._saved_tensor["input"].T.shape)
        self.grad_b = np.sum(grad_output,axis = 0)
        output = np.dot(grad_output,self.W.T)
        return output

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
