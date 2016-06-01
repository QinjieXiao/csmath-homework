


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import leastsq




pi = np.pi

def sin(x):
    return np.sin(x)

def sample(x):
    samples = sin(x);
    samples = [ i + random.gauss(0,0.1) for i in  samples]
    return samples



def draw(x,w,title = "default"):
    plt.title(title)
    #draw sin(x)
    plt.xlabel('x')
    plt.ylabel('y=sin(x)')
    x_for_sin = np.linspace(0,2*pi,1000)
    y_from_sin = sin(x_for_sin)
    plt.plot(x_for_sin, y_from_sin, label= 'y=sin(x)',color = 'g')

    #draw sample points
    plt.scatter(x,sample(x),label = 'with Guass noise')
    #draw fitting curve
    fit_y = np.polyval(w,x_for_sin)
    plt.plot(x_for_sin, fit_y, label='fitting curve', color='r')
    plt.legend()
    plt.show()





x = np.linspace(0,2*pi,10)
y = sample(x)
w = np.polyfit(x,y,3)
draw(x,w,"degree: 3 , samples: 10")
w = np.polyfit(x,y,9)
draw(x,w,"degree: 9 , samples: 10")




def func(w,y,x):
    error = y - np.polyval(w,x)
    error = np.append(error,np.exp(-5)*w)
    return error

w = leastsq(func,w,args=(y,x))[0]
draw(x,w,"degree: 9 , samples: 10 ,with with regularization")






x = np.linspace(0,2*pi,15)
y = sample(x)
w = np.polyfit(x,y,9)
draw(x,w,"degree: 9 , samples: 15")





x = np.linspace(0,2*pi,100)
y = sample(x)
w = np.polyfit(x,y,9)
draw(x,w,"degree: 9 , samples: 100")


