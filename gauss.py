import numpy as np

import math
import pylab
import matplotlib.pyplot as plt

def plt_gauss(sigma=1.0, mean=0.0 ):
	xs = np.arange(-5, 7 , 0.1)
	pts = np.zeros((len(xs),2))
	index = 0
	for x in xs:
		y=1/(sigma*math.sqrt(2*math.pi))*math.exp(-0.5*((x-mean)/sigma)**2)
		pts[index,0], pts[index,1] = x,y/0.4
		index+=1

	return pts

	




data1 = plt_gauss()
data2 = plt_gauss(mean=2)
plt.plot(data1[:,0], data1[:,1], label="Density of Class 1")
plt.plot(data2[:,0], data2[:,1], 'r-', label="Denisty of Class 2")

ax = plt.plot([-1, 2],[0.1, 0.1], 'g.', label="Sample Datapoints", marker='o', markersize=10 )
plt.tick_params( axis='x',  which='both', bottom='off', top='off', labelbottom='off'  )
plt.legend(fontsize=10)
#plt.show()

fig = plt.gcf()
fig.savefig("denisty_estimation1.png")

