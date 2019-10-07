from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss,SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt



train_data, test_data, train_label, test_label = load_mnist_2d('data')
model = []
for i in range(0,4):
    mmodel = Network()
    if i == 1:
        mmodel.add(Linear("Linear1", 784, 256, 0.01))
        mmodel.add(Relu("Relu0"))
        mmodel.add(Linear("Linear2", 256, 10, 0.01))
        mmodel.add(Relu("Relu"))
    model.append(mmodel)

# Your model defintion here
# You should explore different model architecture
#model = Network()
#model.add(Linear('linear1', 784, 200, 0.01))
#model.add(Linear("Linear1", 784, 10, 0.01))
#model.add(Sigmoid("Relu"))
#model.add(Relu('relu1'))
#model.add(Linear('linear2',200,64,0.01))
#model.add(Relu('relu2'))
#model.add(Linear('linear3',64,10,0.01))

#loss = SoftmaxCrossEntropyLoss("loss")
#loss = EuclideanLoss("loss")
#loss = (name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.001,
    'weight_decay': 0.0,
    'momentum': 0.5,
    'batch_size': 50,
    'max_epoch': 1000,
    'disp_freq': 50,
    'test_epoch': 5
}

train_loss = {}
train_acc = {}
test_loss = {}
test_acc = {}

for i in range(0,4):
    if i == 1:
        train_acc.update({1:[]})
        train_loss.update({1:[]})
        test_acc.update({1:[]})
        test_loss.update({1:[]})
        loss = EuclideanLoss("loss")
        for epoch in range(config['max_epoch']):
            LOG_INFO('Training @ %d epoch...' % (epoch))
            result = train_net(model[i], loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
            train_acc[1].append(result["acc"])
            train_loss[1].append(result["loss"])

            if epoch % config['test_epoch'] == 0:
                LOG_INFO('Testing @ %d epoch...' % (epoch))
                result = test_net(model[i], loss, test_data, test_label, config['batch_size'])
                test_loss[1].append(result["loss"])
                test_acc[1].append(result["acc"])

    