# SingeLayerPerceptron
TUGAS 1 MACHINE LEARNING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Insert Theta, Bias, Alpha, Epoch first
theta = np.array ([0.8, 0.287, 0.363, 0.439])
bias = np.array ([0.6])
alpha = 0.8
epoch = 60
localerror = 0.000000000000000000
error = np.zeros([epoch])

#Import
df = pd.read_csv('E:/semester 6/ml/tugas/tugas1/code/uci.data', header=None, nrows=100)
#df[df[4]=='*']
#df[4].loc[pd.to_numeric(df[4], errors='coerce').isnull()]

df[4] = df[4].apply(lambda x:str(x).replace('Iris-setosa', '1'))
df[4] = df[4].apply(lambda x:str(x).replace('Iris-versicolor', '0'))
df[4] = pd.to_numeric(df[4])
#print(df[4])

#Function
#h(x, theta, bias) = theta1*x1 + theta2*x2 + theta3*x3 + theta4*x4 + bias
def hFunc(x, theta, bias):
    sum = bias.copy()
    for i in range(len(x)):
        sum += [x[i]*theta[i]]
    return sum

def sigmo(hx):
    return (1 / (1+math.exp(-1*hx)))
#print(df)

def pred(g):
    if g > 0.5:
        return 0
    else: 
        return 1

def localerr(a,b):
    return math.fabs((a-b))

def delta(g, y, x):
    return (2*(g-y)*(1-y)*g*x)

#Repetition until 60 epoch
for n in range(epoch):
    totalerror = 0
    for i in range(len(df[0])):
        
        x = np.array(df.iloc[i,0:4])
        h = hFunc(x,theta,bias)
        sigmoid = sigmo(h)
        prediction = pred(sigmoid)
        fact = df.iloc[i,4]
        localerror = localerr(sigmoid, fact)

        delta_t = np.zeros(4)
        delta_b = np.zeros(1)

        for j in range(len(delta_t)):
            delta_t[j] = delta(sigmoid, fact, df.iloc[i,j])
            delta_b = np.array(delta(sigmoid, fact, 1))
        
        for k in range(len(theta)):
            theta[k] = theta[k] - (alpha*delta_t[k])
            
        bias = bias - (alpha*delta_b)
        
    #Hitung jumlah error untuk data pada epoch 60
    totalerror = totalerror + localerror

error[n] = totalerror
print("error[", n, "]: ", error[n])


#Plotting data to get the graphy using matplotlib
plt.clf()

x = np.arange(epoch)
y = error.copy()

plt.plot(x,y)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error Graph In Every Epoch")
plt.show
