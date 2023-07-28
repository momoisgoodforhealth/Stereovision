import numpy as np
import matplotlib.pyplot as plt

x=np.array([1000, 1270,700,800])
y=np.array([333, 630, 450,300])
e=np.array([180,210,200,120])
#plt.plot([1000, 1270,700,800], [333, 630, 450,300], 'ro')
plt.ylabel('measured values (mm)')
plt.xlabel('true values (mm)')
plt.title('Distance from Camera')
plt.errorbar(x, y, e, linestyle='None', marker='X')
plt.show()