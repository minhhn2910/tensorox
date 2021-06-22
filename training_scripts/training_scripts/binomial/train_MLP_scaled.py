from sklearn.neural_network import MLPRegressor
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import *
scaler = StandardScaler() #RobustScaler() 0.05
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
#2 input 3 output
x = np.loadtxt('input_noprocess.txt',delimiter=',').reshape(-1,5)
y = np.loadtxt('output_noprocess.txt',delimiter=',').reshape(-1,1)
newlen = int(len(x)/3)*3
#print (newlen)
#print (y[:10,])
x = x[:newlen,:]
y = y[:newlen,:]
#y = y
mean_y = np.sum(y)/len(y)
#print('mean_y ',mean_y)
#y[y == 0.0] = -mean_y
#y[y == 0.0] = -1
x = x.flatten().reshape(-1,15)
y = y.flatten().reshape(-1,3)
print ("sanity check \n input ", x[0], "\n output ", y[0])
x_train = x#[:-2000,:]
y_train = np.copy(y)#[:-2000,:]
y_train[y_train<0.0001]=-1
print("xshape ", x_train.shape)
print("yshape ",y_train.shape)
scaler.fit(x_train)
print ("scaler parameter " )
print (scaler.mean_)
print (scaler.scale_)
print ("prescale: ")
print (x_train[0])
x_train = scaler.transform(x_train)
print (x_train[0])
nn = MLPRegressor(hidden_layer_sizes=(16,16),
                  activation='relu',solver='adam', tol=0.00001,verbose=True, max_iter=1000 , n_iter_no_change=20 )

np.random.seed(0)
n = nn.fit(x_train, y_train)
import pickle
with open('scaled3elem.weights', 'wb') as fp:
    pickle.dump(nn.coefs_, fp)
with open('scaled3elem.bias', 'wb') as fp:
    pickle.dump(nn.intercepts_, fp)


x_test = x#[:16,:]
x_test = scaler.transform(x_test)
y_ref = y#[:16]
y_test = nn.predict(x_test)
y_test[y_test<0]=0
print ("result ")
print (y_ref[:10] - y_test[:10])
print ("L1 norm ")
print (sum(np.abs(y_ref.flatten()-y_test.flatten()))/sum(np.abs(y_test.flatten())))

#plt.hist(y_ref- y_test)
#plt.show()

'''
import pickle

weights = pickle.load( open( "blackschole2input.weights", "rb" ))
bias = pickle.load( open( "blackschole2input.bias", "rb" ))
##todo : implement mlp from scratch & use the trained weight for inference.
def activation(x):
    return 1.0/(1.0+np.exp(-x))

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def matmul16    (A, B):
    print ("multiplying matrix shapes ", A.shape, " ", B.shape)
    return np.matmul(A,B)

def mlp_layers(weights, bias, input):
    print ("doing layer1")
    print ("shapes ", weights[0].shape, bias[0].shape)
    print (input.shape)
    output1 = activation(input@weights[0] + bias[0])
    #print (input@weights[0])
    output2 = activation(output1@weights[1] + bias[1])
    #print (output2)
    output  = output2@weights[2] + bias[2]
    #print (output)
    return output2@weights[2] + bias[2]

np.random.seed(1)

test_x = (np.random.uniform(0,100,size=32)).reshape(2, -1)
row = np.array([[0]*16])
print (test_x.shape[0])
print (row.shape)
test_x_padded = test_x
for i in range(test_x.shape[0], 16):
    test_x_padded = np.concatenate((test_x_padded,row), axis=0)
#print (test_x_padded)
ref_y =  np.array(list(map(callPrice,test_x[0], test_x[1])))
test_x = test_x.T
print ("ref_y")
#print (ref_y)
print (weights[0].shape)
weights_paded = weights[0]
for i in range(weights[0].shape[0], 16):
    weights_paded = np.concatenate((weights_paded,row), axis=0)

weights_paded_2 = weights[2].T
for i in range(1, 16):
    weights_paded_2 = np.concatenate((weights_paded_2,row), axis=0)
weights_paded_2 = weights_paded_2.T
output1 = mlp_layers(weights, bias, test_x)
print ("---------")
matmul_res = activation(matmul16(test_x_padded.T,weights_paded) + bias[0])
matmul_res = activation(matmul16(matmul_res,weights[1]) + bias[1])
matmul_res = matmul16(matmul_res,weights_paded_2) + bias[2]
matmul_res = matmul_res.T[0]
print (" -------- ")

#print (matmul16(test_x_padded.T,weights_paded))
#print (bias[0])
#print (matmul16(test_x_padded.T,weights_paded)+ np.array([bias[0]]*16))
print (output1)
print (matmul_res)
print (weights[2].shape)
'''
