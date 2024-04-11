import numpy as np
import torch
import time
import platform


print(f'Pytorch version: {torch.__version__}')
print(f'cuda version: {torch.version.cuda}')
print(f'Python version: {platform.python_version()}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

from MNISTtools import load, show

xtrain, ltrain = load(dataset='training', path='data/')
xtest, ltest = load(dataset='testing', path='data/')

def normalize_MNIST_images(x):
    '''
    Args:
        x: data
    '''
    x_norm = x.astype(np.float32)
    return x_norm*2/255-1


# normalization
xtrain = normalize_MNIST_images(xtrain)
xtest = normalize_MNIST_images(xtest)

print("RESHAPE")
# reshape to 3d
xtrain = xtrain.reshape([28,28,-1])[:,:,None,:]
xtest = xtest.reshape([28,28,-1])[:,:,None,:]
print(f'shape of xtrain after reshape is {xtrain.shape}.')
print(f'shape of xtest after reshape is {xtest.shape}.')
print(" ")

print("MOVE AXIS")
# moveaxis
xtrain = np.moveaxis(xtrain, (2,3), (1,0))
xtest = np.moveaxis(xtest, (2,3), (1,0))
print(f'shape of xtrain after moveaxis is {xtrain.shape}.')
print(f'shape of xtest after moveaxis is {xtest.shape}.')
print(" ")

xtrain = torch.from_numpy(xtrain)
ltrain = torch.from_numpy(ltrain)
xtest = torch.from_numpy(xtest)
ltest = torch.from_numpy(ltest)


xtrain_gpu = xtrain.to(device)
ltrain_gpu = ltrain.to(device)
xtest_gpu = xtest.to(device)
ltest_gpu = ltest.to(device)


import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)


net = LeNet()
print(net)

print("TEST WITHOUT GRADIENT TRACKING")
for name, param in net.named_parameters():
    print(name, param.size(), param.requires_grad)

# avoid tracking for gradient during testing and then save some computation time
with torch.no_grad():
    yinit = net(xtest)

_, lpred = yinit.max(1)
print(100 * (ltest == lpred).float().mean())
print(" ")

def backprop_deep(xtrain, ltrain, net, T, B=100, gamma=.001, rho=.9):
    '''
    Backprop.
    
    Args:
        xtrain: training samples
        ltrain: testing samples
        net: neural network
        T: number of epochs
        B: minibatch size
        gamma: step size
        rho: momentum
    '''
    N = xtrain.size()[0]     # Training set size
    NB = N//B                # Number of minibatches
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=gamma, momentum=rho)
    
    for epoch in range(T):
        running_loss = 0.0
        shuffled_indices = np.random.permutation(NB)
        for k in range(NB):
            # Extract k-th minibatch from xtrain and ltrain
            minibatch_indices = range(shuffled_indices[k]*B, (shuffled_indices[k]+1)*B)
            inputs = xtrain[minibatch_indices]
            labels = ltrain[minibatch_indices]

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)

            # Error evaluation
            loss = criterion(outputs, labels)

            # Back propagation
            loss.backward()

            # Parameter update
            optimizer.step()

            # Print averaged loss per minibatch every 100 mini-batches
            # Compute and print statistics
            with torch.no_grad():
                running_loss += loss.item()
            if k % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, k + 1, running_loss / 100))
                running_loss = 0.0


net = LeNet()
print("TRAIN EPOCH 3")
start = time.time()
backprop_deep(xtrain, ltrain, net, T=3)
end = time.time()
print(f'It takes {end-start:.6f} seconds.')
print(" ")

print("TEST EPOCH 3")

start = time.time()
backprop_deep(xtest, ltest, net, T=3)
end = time.time()
print(f'It takes {end-start:.6f} seconds.')
print(" ")

y = net(xtest)
print(100 * (ltest==y.max(1)[1]).float().mean())
print(" ")

print("TRAIN EPOCH 10")
net_gpu = LeNet().to(device)
start = time.time()
backprop_deep(xtrain_gpu, ltrain_gpu, net_gpu, T=10)
end = time.time()
print(f'It takes {end-start:.6f} seconds.')


y = net_gpu(xtest_gpu)
print(100 * (ltest==y.max(1)[1].cpu()).float().mean())
