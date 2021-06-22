import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.preprocessing import StandardScaler
# Device configuration
device = 'cpu'#torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
scaler = StandardScaler()
# Hyper parameters
num_epochs = 25
batch_size = 200
learning_rate = 0.001

# dataset

x = np.loadtxt('input.txt',delimiter =",",dtype=np.float32).flatten().reshape(-1,27)
y = np.loadtxt('output.txt',delimiter =",",dtype=np.float32).flatten().reshape(-1,3)
newlen = len(x)#int(len(x)/3)*3
print (newlen)
#print (y[:10,])
x = x[:newlen,:]
y = y[:newlen,:]
y = y
#x = x.flatten().reshape(-1,12)
#y = y.flatten().reshape(-1,3)
print ("sanity check \n input ", x[0], "\n output ", y[0])
x_train = x[:-1000,:]
y_train = y[:-1000,:]

print("xshape ", x_train.shape)
print("yshape ",y_train.shape)
scaler.fit(x_train)
print ("scaler parameter " )
print (scaler.mean_)
print (scaler.scale_)
print ("prescale: ")
print (x_train[0])
x_train = scaler.transform(x_train)
print ("afterscale: ")
print (x_train[0])
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass
    def __getitem__(self, index):
        return (x_train[index], y_train[index])#.reshape(16,16))
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return y_train.shape[0]
        #return 0
# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc0 = nn.Linear(27,8)
        self.fc1 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)


    def forward(self, x):
        out = self.fc0(x)
        out = self.relu(out)
#        out = self.fc1(out)
#        out = self.relu(out)
        out = self.fc1(out)
        return out
model = NeuralNet().to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay = 0.0001)
train_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
total_step = len(train_loader)
#test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                   batch_size=batch_size,
#                                      shuffle=False)

running_loss = 0.0
smallest_loss = 1e10
smallest_epoch = 0
# Train the model
for epoch in range(num_epochs):
    break_cont = False
    for i, (inputs, targets) in enumerate(train_loader):
        # Move tensors to the configured device
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % total_step == total_step-1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / total_step))
            if (running_loss / total_step < smallest_loss):
                smallest_loss = running_loss / total_step
                smallest_epoch = epoch
            else:
                if(epoch - smallest_epoch > 10):
                    break_cont = True
            running_loss = 0.0
    if (break_cont):
        break

torch.save(model.state_dict(), 'model.ckpt')
predicted = model(torch.from_numpy(x_train)).detach().cpu().numpy()
print(sum(np.abs(predicted.flatten() - y_train.flatten())) / len(predicted.flatten()))
print(sum(np.abs(predicted.flatten() - y_train.flatten())) / sum(y_train.flatten()))
print ("avg rel err")
new_train = y_train.flatten()
new_train[new_train == 0] = 1
print (sum(np.abs(predicted.flatten() - y_train.flatten())/np.abs(new_train)) / len(predicted.flatten()))
print(y_train[:16])
print ("predict")
print(predicted[:16])

# Save the model checkpoint
