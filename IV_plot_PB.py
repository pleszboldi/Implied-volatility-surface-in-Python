import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd

#adatok betoltese
path="C:/d"
X=np.load(path+'dms0.npy',allow_pickle=True)
Y=np.load(path+'dms1.npy',allow_pickle=True)
Z=np.load(path+'dms2.npy',allow_pickle=True)
X=np.array(X,dtype=np.float64)
Y=np.array(Y,dtype=np.float64)

#2D plot
mycolor=(150/255,5/255,160/255)
#adott strike
k=10 #strike-ok vektorabol valamelyik
y=Z[k,:]
x=Y
figure,axes=plt.subplots()
axes.plot(x,y,color=mycolor)
axes.set(xlabel='Lejáratok',ylabel='Implied volatility',title='Implied volatility lejáratok függvényében')

#adott lejarat
t=10 #lejaratok vektorabol valamelyik
y=Z[:,t]
x=X
figure,axes=plt.subplots()
axes.plot(x,y,color=mycolor)
axes.set(xlabel='Strikeok',ylabel='Implied volatility',title='Implied volatility strikeok függvényében')

#3D plot
X, Y = np.meshgrid(Y, X)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

