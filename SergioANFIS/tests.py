from anfis_model import ANFIS
import membershipfunction
import numpy as np


file_path = "trainingSet.txt"
try:
    ts = np.loadtxt(file_path, usecols=[1,2,3])
except Exception as e:
    print(f"Error loading file: {e}")
    exit()


X = ts[:,0:2]
Y = ts[:,2]

mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]

# Updating the model with MFs
mfc = membershipfunction.MemFuncs(mf)

# Creating the ANFIS Model Object
anf = ANFIS(X, Y, mfc)

# Plot the MFs pre-training
anf.plotMF(X[:,0],0)
anf.plotMF(X[:,1],1)

# Fitting the ANFIS Model
anf.trainHybridJangOffLine(epochs=20, k=0.01) # k: learning rate

# Print RMSE
rmse = np.sqrt((np.square(np.subtract(anf.fittedValues, anf.Y)).mean()))
print('RMSE:', rmse)

# Plotting Model performance
anf.plotErrors()
anf.plotResults()

# Plot the MFs post-training
anf.plotMF(X[:,0],0)
anf.plotMF(X[:,1],1)
