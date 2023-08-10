import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x1=pd.Series(data=[0.94,0.93,0.93])

x2=pd.Series(data=[1.6,1.57,1.56])
x3=pd.Series(data=[1.83,1.76,1.74])
x4=pd.Series(data=[2.14,2.1,2.09])
x5=pd.Series(data=[2.23,2.21,2.22])
x6=pd.Series(data=[2.4,2.38,2.36])
x7=pd.Series(data=[2.58,2.6,2.56])
x=np.array([x1.mean(),x2.mean(),x3.mean(),x4.mean(),x5.mean(),x6.mean(),x7.mean()])
e=np.array([x1.std(),x2.std(),x3.std(),x4.std(),x5.std(),x6.std(),x7.std()])
y=np.array([0.5,1,1.5,2,2.5,3,3.5])
print(x)
print(e)
plt.errorbar(x=x, y=y, xerr=e, linestyle='None', marker='.')

plt.ylabel('True Value (meters)')
plt.xlabel('Measured Value (meters)')
plt.title('Different Objects at Same Distance')
plt.savefig('stdgraph.png', dpi=500)
#plt.legend( loc="lower right")
#plt.savefig('woline.png', dpi=500)
#plt.errorbar(x, y, e, linestyle='None', )
plt.show()