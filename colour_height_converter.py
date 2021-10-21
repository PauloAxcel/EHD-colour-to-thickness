# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 11:11:40 2021

@author: paulo
"""

from PIL import Image
import numpy as np
from collections import namedtuple, OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pylab
import scipy as sp
Polynomial = np.polynomial.Polynomial
from matplotlib.pyplot import figure
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC   
import numpy as np
import cv2
import matplotlib.pyplot as plt


############## CALIBRATION TO GET RGB TO TICKNESS CONVERTION

font = {'family' : 'Arial',
        'size'   : 22}

plt.rc('font', **font)

img = cv2.imread('calibration.png')
cal =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#cal1d = cal[cal.shape[0]//2,:,:]
cal1d = np.mean(cal,axis=0,dtype=int)
#stdcal1d = np.max(np.std(cal,axis=0),axis=1)
stdcal1d = 5
#tabulated values from Erik Schaffer

thickness=[20,70,100,140,200,265,290,310,330,350,400,420,460,520]
Rval = [154,98,43,47,253,176,78,88,168,244,252,158,228,247]
Gval = [86,58,76,117,231,76,166,207,243,247,130,223,207,174]
Bval = [15,48,107,143,119,130,176,175,140,94,154,123,128,133]



def indexation(cal1d,stdcal1d,Rval,Gval,Bval):
    counter = 0    
    x = []
    xdim = np.linspace(0,1,len(cal1d))
    
    for r,g,b in zip(Rval,Gval,Bval):
        
    
        Rmin = r - stdcal1d < cal1d[:,0]
        Rmax = cal1d[:,0] < r+ stdcal1d
        
        Gmin = g - stdcal1d < cal1d[:,1]
        Gmax = cal1d[:,1] < g + stdcal1d
        
        Bmin = b - stdcal1d < cal1d[:,2]
        Bmax = cal1d[:,2] < b+stdcal1d
        
        #plt.plot(xdim,Rmin,'r')
        #plt.plot(xdim,Rmax,'--r')
        #plt.plot(xdim,Rmax&Rmin,'r')
        #plt.plot(xdim,Gmin+1.1,'g')
        #plt.plot(xdim,Gmax+1.1,'--g')
        #plt.plot(xdim,Gmax&Gmin,'g')
        #plt.plot(xdim,Bmin+2.2,'b')
        #plt.plot(xdim,Bmax+2.2,'--b')
        #plt.plot(xdim,Rmax&Rmin&Gmax&Gmin&Bmin&Bmax,'k')
        
        index = list(Rmax & Rmin & Gmax & Gmin & Bmin & Bmax)
        
        position = np.mean([index for index, element in enumerate(index) if element == True],dtype=int)

        
#        plt.figure(figsize=(9,9/1.618))
#        plt.plot(xdim,cal1d[:,0],color='r',label='red')
#        plt.plot(xdim,cal1d[:,1],color='g',label='green')
#        plt.plot(xdim,cal1d[:,2],color='b',label='blue')
#        plt.hlines(r,0,1,color='r',ls='dashed')
#        plt.hlines(g,0,1,color='g',ls='dashed')
#        plt.hlines(b,0,1,color='b',ls='dashed')
#        plt.vlines(xdim[position],0,255,color='k',ls='dashed')
        
        if position<len(cal1d)+1 and position>0:
            x.append(xdim[position])
        else:
            x.append(0)
            counter +=1
    
    
    return x,counter
    


from scipy.optimize import curve_fit

def objective(x, a, b):
	return a * x + b


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = sp.stats.linregress(x, y)
    return r_value**2


x,counter = indexation(cal1d,stdcal1d,Rval,Gval,Bval)

popt, _ = curve_fit(objective, x, thickness)

a, b = popt

y_line = [objective(xs, a, b) for xs in x]

plt.figure(figsize=(9,9/1.618))
plt.plot(x,thickness,'o')
plt.plot(x, y_line, '--', color='red',label='$R^2$='+str(round(rsquared(x,thickness),2))+'\n thickness(nm)='+str(round(a,2))+'*RGB'+str(round(b,2)))
plt.legend(loc='best')
plt.show()

#### CASE STUDY TO MAP RGB INTO HEIGHT

img2 = cv2.imread('test_image.png')
test =  cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
tnew = np.zeros((test.shape[0],test.shape[1]))

#from skimage.filters import threshold_minimum
#thresh_min = threshold_minimum(test)
#binary_min = test > thresh_min

#plt.hist(test[:,:,0].ravel(), bins=256,color='r')
#plt.hist(test[:,:,1].ravel(), bins=256,color='g')
#plt.hist(test[:,:,2].ravel(), bins=256,color='b')

Rnew = test[:,:,0]
Gnew = test[:,:,1]
Bnew = test[:,:,2]

cnt = []
filt = 25

for i in range(Rnew.shape[0]):
    print(i)
    xnew,counter = indexation(cal1d,filt,list(Rnew[i,:]),list(Gnew[i,:]),list(Bnew[i,:]))
    tnew[i,:] = [a*xnew2+b for xnew2 in xnew]
    cnt.append(counter)
    


fig = plt.figure()
ax = fig.add_subplot(121)
im = ax.imshow(tnew)
#ax.clim(0, tnew.max()-tnew.max()%50+50)
fig.colorbar(im)

ax2 = fig.add_subplot(122)
ax2.imshow(test)


#plt.figure()
#plt.plot(filt,np.mean(counter))


xx = np.linspace(0,5,tnew.shape[1])
yy = np.linspace(0,5,tnew.shape[0])

X,Y = np.meshgrid(xx,yy)

Z = tnew

from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.plot_surface(X, Y, tnew)
surf = ax.plot_surface(X, Y, -Z, cmap=cm.coolwarm,  linewidth=0, antialiased=False)
ax.view_init(elev=50, azim=-180)


# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
#ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

#plt.show()







