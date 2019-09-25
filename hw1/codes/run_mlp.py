from network import Network
from utils import LOG_INFO
from layers import Relu, Sigmoid, Linear
from loss import EuclideanLoss,SoftmaxCrossEntropyLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt



train_data, test_data, train_label, test_label = load_mnist_2d('data')
model = []
for i in range(0,7):
    mmodel = Network()
    mmodel.add(Linear("Linear1", 784, 10, 0.01))
    mmodel.add(Sigmoid("Relu"))
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

loss = SoftmaxCrossEntropyLoss("loss")
#loss = EuclideanLoss("loss")
#loss = (name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.01,
    'weight_decay': 0.0,
    'momentum': 0.75,
    'batch_size': 50,
    'max_epoch': 1000,
    'disp_freq': 50,
    'test_epoch': 5
}

train_loss = {}
train_acc = {}
test_loss = {}
test_acc = {}

for i in range(0,7):
    value = config["weight_decay"]
    if value < 0.00001:
        config['weight_decay'] += 0.01
    elif value < 0.09:
        config['weight_decay'] *= 10
    else:
        if value < 0.2:
            config['weight_decay'] = 0.3
        elif value < 0.4:
            config['weight_decay'] = 0.5
        elif value < 0.6:
            config['weight_decay'] = 0.7
        else:
            config['weight_decay'] = 0.9

    if i == 0:
        config['weight_decay'] = 0.0

    train_acc.update({config['weight_decay']:[]})
    train_loss.update({config['weight_decay']:[]})
    test_acc.update({config['weight_decay']:[]})
    test_loss.update({config['weight_decay']:[]})
    for epoch in range(config['max_epoch']):
        LOG_INFO('Training @ %d epoch...' % (epoch))
        result = train_net(model[i], loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])
        train_acc[config['weight_decay']].append(result["acc"])
        train_loss[config['weight_decay']].append(result["loss"])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % (epoch))
            result = test_net(model[i], loss, test_data, test_label, config['batch_size'])
            test_loss[config['weight_decay']].append(result["loss"])
            test_acc[config['weight_decay']].append(result["acc"])


plt.figure()

plt.subplot(221)
plt.title("weight_decay")
plt.plot(train_loss[0.0], color = 'pink', label = '0.0')
plt.plot(train_loss[0.01], color = 'green', label = "0.01")
plt.plot(train_loss[0.1], color = 'red', label = "0.1")
plt.plot(train_loss[0.3], color = 'skyblue', label = "0.3" )
plt.plot(train_loss[0.5], color = 'blue', label = "0.5" )
plt.plot(train_loss[0.7], color = 'yellow', label = "0.7" )
plt.plot(train_loss[0.9], color = 'purple', label = "0.9" )
plt.legend()
plt.xlabel('iteration')
plt.ylabel('train_loss')

plt.subplot(222)
plt.title("weight_decay")
plt.plot(train_acc[0.0], color = 'pink', label = '0.0')
plt.plot(train_acc[0.01], color = 'green', label = "0.01")
plt.plot(train_acc[0.1], color = 'red', label = "0.1")
plt.plot(train_acc[0.3], color = 'skyblue', label = "0.3" )
plt.plot(train_acc[0.5], color = 'blue', label = "0.5" )
plt.plot(train_acc[0.7], color = 'yellow', label = "0.7" )
plt.plot(train_acc[0.9], color = 'purple', label = "0.9" )
plt.legend()
plt.xlabel('iteration')
plt.ylabel('train_acc')

plt.subplot(223)
plt.title("weight_decay")
plt.plot(test_loss[0.0], color = 'pink', label = '0.0')
plt.plot(test_loss[0.01], color = 'green', label = "0.01")
plt.plot(test_loss[0.1], color = 'red', label = "0.1")
plt.plot(test_loss[0.3], color = 'skyblue', label = "0.3" )
plt.plot(test_loss[0.5], color = 'blue', label = "0.5" )
plt.plot(test_loss[0.7], color = 'yellow', label = "0.7" )
plt.plot(test_loss[0.9], color = 'purple', label = "0.9" )
plt.legend()
plt.xlabel('iteration')
plt.ylabel('test_loss')

plt.subplot(224)
plt.title("weight_decay")
plt.plot(test_acc[0.0], color = 'pink', label = '0.0')
plt.plot(test_acc[0.01], color = 'green', label = "0.01")
plt.plot(test_acc[0.1], color = 'red', label = "0.1")
plt.plot(test_acc[0.3], color = 'skyblue', label = "0.3" )
plt.plot(test_acc[0.5], color = 'blue', label = "0.5" )
plt.plot(test_acc[0.7], color = 'yellow', label = "0.7" )
plt.plot(test_acc[0.9], color = 'purple', label = "0.9" )
plt.legend()
plt.xlabel('iteration')
plt.ylabel('test_acc')

plt.show()
