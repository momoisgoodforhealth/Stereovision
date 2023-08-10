import numpy as np
import matplotlib.pyplot as plt

x=np.array([0.7, 1.15,1.3, 1.41,1.75, 1.82,1.99,2,2.05,2.4,2.7,2.9, 3.1,3.21,4.67,5.2])


y=np.array([1.069,1.56, 1.7,1.49,1.76,1.77,1.99,2.05,2.02,2.2,2.3,2.7,2.3,2.4,2.44,2.7])
z=np.array([1280,890,1023,938,790,770,700,690,680,637,625,520,594,582,580,500])
e=np.array([180,210,200,120])
#plt.plot([1000, 1270,700,800], [333, 630, 450,300], 'ro')
plt.plot(x, z, color='r', marker=".",label='True Value',linestyle = 'None')
plt.plot(y, z, color='g', marker="x",label='Measured Value',linestyle = 'None')
model3 = np.poly1d(np.polyfit(x, z, 5))
polyline = np.linspace(0, 5, 5)
#plt.plot(polyline, model3(polyline), color='red')
plt.ylabel('Disparity')
plt.xlabel('Distance from Camera (meters)')
plt.title('Disparity - Distance Graph')
plt.legend( loc="upper right")
plt.savefig('woline.png', dpi=500)
#plt.errorbar(x, y, e, linestyle='None', )
plt.show()