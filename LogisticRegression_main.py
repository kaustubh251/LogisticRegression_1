import matplotlib as mplot
import numpy as npy
import statistics as stats
file = open("ex2data1.txt","rt")
data = file.read()
data1 = data.split()
x = []
y = []
count = 0
for element in data1:
    datapoint = element.split(',')
    m = len(datapoint)
    element1 = [1]
    for i in range(m-1):
        element1.append(float(datapoint[i]))
    x.insert(count, element1)
    y.append(int(datapoint[m-1]))
    count += 1

def H(z):
    g = 1/(1 + npy.exp(-z))
    return g

def Z(theta, x1):
    z = 0
    count = 0
    for element in theta:
        z += element*x1[count]
        count += 1
    return z

def cost(theta, x, y):
    cost = 0
    m = len(x)
    for i in range(m):
        x1 = x[i][:]
        z = Z(theta, x1)
        h = H(z)
        cost += -(y[i]*npy.log(h) + ((1-y[i])*npy.log(1-h)))
    cost = cost/m
    return cost

def gradDescent(theta, x, y, alpha, maxIter):
    for k in range(maxIter):
        theta1 = theta
        for i in range(len(theta)):
            derCost = 0
            j = 0
            for j in range(len(x)):
                x1 = x[j][:]
                z = Z(theta, x1)
                h = H(z)
                derCost += (h - y[j])*x[j][i]
            theta1[i] = theta[i] - alpha*derCost/len(x)      
        theta = theta1
    return theta

print("Enter 3 parameters one by one:")
init_Theta = []
init_Theta.append(float(input()))
init_Theta.append(float(input()))
init_Theta.append(float(input()))
print("Enter 2 datapoints:")
init_Datapoint = [1]
init_Datapoint.append(float(input()))
init_Datapoint.append(float(input()))
print("Enter learning rate:")
alpha = float(input())
print("Enter maximum number of iterations for Gradient Descent to converge:")
maxIter = int(input())
theta_Final = gradDescent(init_Theta, x, y, alpha, maxIter)
z = Z(theta_Final, init_Datapoint)
h = H(z)
print("The probability of student to get admission in the college is:")
print(h)