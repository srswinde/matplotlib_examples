import matplotlib.pyplot as plt
import numpy as np


n=200
data = [ np.random.normal(0,1.0) for x in range(n) ]


plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off',
    left='off',
    labelleft='off')


plt.hist(data, n/10, alpha=0.5, color="#e0000e", normed=True, label="Histogram p(x)")
plt.plot(data, np.zeros((len(data)))+0.80, 'x', color="#000000", fillstyle='none', label="data set" )
plt.legend()
plt.show()

