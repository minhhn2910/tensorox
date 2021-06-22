from sklearn.neural_network import MLPRegressor
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

weights = pickle.load( open( "invk2j.weights", "rb" ))
bias = pickle.load( open( "invk2j.bias", "rb" ))
np.set_printoptions(precision=4)
def activation(x):
    if (x < 0):
        return 0
    else:
        return x
#    return 1.0/(1.0+np.exp(-x))
prettylist2g = lambda l : '[%s]' % ', '.join("%.6g" % x for x in l)
#activation_vec = np.vectorize(activation)
def activation_vec(x):
    temp  = x.flatten()
    for i in range(len(temp)):
        if (temp[i]<0):
            temp[i]= 0.0
    return temp.reshape(x.shape)


def matmul16    (A, B):
    print ("multiplying matrix shapes ", A.shape, " ", B.shape)
    return np.matmul(A,B)

def mlp_layers(weights, bias, input):
    print ("doing layer1")
    print ("shapes ", weights[0].shape, bias[0].shape)
    print (input.shape)
    output1 = activation_vec(input@weights[0] + bias[0])
    output2 = activation_vec(output1@weights[1] + bias[1])

    output  = output2@weights[2] + bias[2]
    #print (weights[2] )
    return output#output2@weights[2] + bias[2]

x = np.loadtxt('input_invk2j.txt',delimiter=',').reshape(-1,2)
y = np.loadtxt('output_invk2j.txt',delimiter=',').reshape(-1,3)
newlen = int(len(x)/2)*2
print (newlen)
x = x[:newlen,:]
y = y[:newlen,:]
x = x.flatten().reshape(-1,4)
y = y.flatten().reshape(-1,6)
scaler = StandardScaler()
scaler.fit(x)
print ("scaler parameter " )
print (scaler.mean_)
print (1.0/scaler.scale_)
np.random.seed(1)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
#test_x = (np.loadtxt("input.txt",delimiter=",")).reshape(-1, 15)
#test_x = test_x[-16:,:]
test_x = x[:16,]#np.array(x_test).reshape(-1,15)
#printf(test_x.shape)
#print ("prescaled ", test_x)
test_x = scaler.transform(test_x);#(test_x - means)/scale

#print ("after scaled " ,test_x)

row = np.array([[0]*16])
#print (test_x)
'''
test_x_padded = test_x.T
for i in range(test_x.shape[1], 16):
    test_x_padded = np.concatenate((test_x_padded,row), axis=0)

test_x_padded = test_x_padded
'''
ref_y = y[:16,]
#test_x = test_x.T
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

print ("test_x--------------")
print (test_x[::1])
print ("test_x--------------")

print ("ref_y--------------")
print (ref_y[::1])
print ("ref_y--------------")
'''
for i in range(3):
    print (' shape ', weights[i].shape, bias[i].shape)
    print ('weight ')
    weight_temp = weights[i].flatten()
    weight_temp[np.abs(weight_temp)<1e-10] = 0.0
    print(prettylist2g(list(weight_temp)))
    print ('bias ')
    print (prettylist2g(list(bias[i].flatten())))

print ("weight 0 ")
print (prettylist2g(list(weights_paded.flatten())))
print ("weight 1")
print (prettylist2g(list(weights[1].flatten())))

weights_paded = weights[0]
for i in range(weights[0].shape[0], 16):
    weights_paded = np.concatenate((weights_paded,row), axis=0)

weights_paded_2 = weights[2].T
for i in range(weights_paded_2.shape[0], 16):
    weights_paded_2 = np.concatenate((weights_paded_2,row), axis=0)
weights_paded_2 = weights_paded_2.T
#print ("weight 2 shape ")
'''
'''
print (weights[2].shape)
print (", ".join("%.8f" % f for f in list(weights_paded_2.flatten())) )

'''
#print ("weight2")print (prettylist2g(list(weights[2].flatten())))
#test_x = test_x.T
#test_x = np.array(x_test).reshape(16,15)

output1 = mlp_layers(weights, bias, test_x)
print ("------output---")
print (output1[::1]*100.0)

#print (weights_paded)
'''
print ("bias 0 ")
print (list(bias[0]))
print ("bias 1" )
print (list(bias[1]))
print ("bias 2")
print (list(bias[2]))


bias2_appended = list(bias[2])
for i in range(len(bias[2]), 16):
	bias2_appended.append(0.0)
#print (bias2_appended)
bias2_appended = np.array(bias2_appended)
#print (list(bias[2]))
res = matmul16(test_x_padded.T,weights_paded)

#print ( list(bias[0]))
#print (matmul16(test_x_padded.T,weights_paded))
matmul_res = activation_vec(matmul16(test_x_padded.T,weights_paded) + bias[0])
#print (matmul16(test_x_padded.T,weights_paded))
print("hidden layer 1")
#print (matmul_res)
matmul_res = activation_vec(matmul16(matmul_res,weights[1]) + bias[1])
print("hidden layer 2")
#print (matmul_res)
#print (matmul_res)
#matmul_res = matmul16(matmul_res,weights_paded_2) + bias[2]
matmul_res = matmul16(matmul_res,weights_paded_2) + bias2_appended
'''

print (" -------- ")

#print (prettylist2g(list(weights_paded_2.flatten())))
#print (matmul16(test_x_padded.T,weights_paded))
#print (bias[0])
#print (matmul16(test_x_padded.T,weights_paded)+ np.array([bias[0]]*16))
#print (output1)
#print (matmul_res)
#print (weights[2].shape)
