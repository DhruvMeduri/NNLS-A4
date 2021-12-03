from PIL import Image
from numpy import asarray
import numpy as np
import matplotlib.pyplot as plt

# load the image and convert into
# numpy array
err = []
img = Image.open('pic.jpg')
learn = 10**(-8)
# asarray() class is used to convert
# PIL images into NumPy arrays
numpydata = asarray(img)/255

# <class 'numpy.ndarray'>
print(type(numpydata))
print(numpydata)

#  shape
print(numpydata.shape)

weights  = np.random.rand(64,8) #64X8 weight matrix
for epoch in range(2000):
    mse = 0
    print(epoch)
    for i in range(0,256,8):
        for j in range(0,256,8):
            input = np.zeros(64)
            for r in range(8):
                for c in range(8):
                    input[8*r + c] = numpydata[i+r][j+c] # input is 1X64 entries
            output = np.dot(input,weights) #1X8 matrix
            # Now to update the weights
            for a in range(64):
                for b in range(8):
                    sum = 0
                    for k in range(b):
                        sum = sum + weights[a][k]*output[k]
                    weights[a][b] = weights[a][b] + learn*(output[b]*input[a] - output[b]*sum)
    #Now to reconstruct the image and calculate the MSE
    temp_rec = np.zeros(64) # 1X64 - reconstructed using the eigen vectors
    for i in range(0,256,8):
        for j in range(0,256,8):
            input = np.zeros(64)
            for r in range(8):
                for c in range(8):
                    input[8*r + c] = numpydata[i+r][j+c] # input is 1X64 entries
            output = np.dot(input,weights) #1X8 matrix
            temp_rec = np.dot(weights,output)
            for l in range(64):
                mse = mse + (input[l] - temp_rec[l])*(input[l] - temp_rec[l])
    print(mse)
    err.append(mse)



print(err)
x_list = []
for i in range(1,2001):
    x_list.append(i)

plt.plot(x_list,err)
plt.xlabel("Scans")
plt.ylabel("MSE")
plt.title("Lena")
plt.show()
