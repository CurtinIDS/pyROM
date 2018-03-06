# pyROM - Reduced Order Modelling Framework for Python

## Background

Solving a PDE numerically for mesh based/time dependent codes usually involves iterating through the whole spatial basis that represents the underlying system. Reduced Order Modelling involves scanning for dominant "modes" that contain most information of the happening sections. Quantitatively, a 10^4x10^4 matrix could be transformed in to a 100x3 matrix, evolution of which (and retransformation back to 10^4x10^4) may result in negligible error.

## Algorithm

1.  Fetch stiffness matrix, snapshot matrix(evolved for Δt timesteps).
2.  Compute PCA/DMD/POD (user choice) to select the first N modes. Ex. - [Φ_n]=[Φ_1,Φ_23,Φ_10]
3.  Solve σ⍺=∂⍺/∂t*, where ⍺ is a kxk matrix(k is min. dimension of [Φ_n], the dependent variable C=[Φ_n][⍺]).
4.  Transform [Φ_n] back to C after t timesteps

*Fetch snapshots as and when needed


## Example

```
#Import pyROM functions
from pyROM import *

#Plotting library
from bokeh.plotting import show, output_notebook


f=lambda x:np.sin(np.float(2*np.pi*x[0]))*np.sin(2*np.pi*x[1]) #perturbation
mydomain=Domain('examples/porous.mat',perturbation=f,cutoff=5,timesteps=24000) #Domain decomposition
mydomain.march() #Run simulation


#POD
mydomain.pod() #Computes POD and stores the value in mydomain.POD

#DMD 
mydomain.dmd() #does DMD and stores the values in mydomain.DMD

#Visualize POD
output_notebook()  #Opens bokeh notebook
out=mydomain.bokeh_out(mydomain,mydomain.POD,timesteps=300)
show(out)

#Visualize DMD
out=mydomain.bokeh_out(mydomain,mydomain.DMD,timesteps=300)
show(out)


#Visualize POD/DMD together
out=mydomain.bokeh_out(mydomain,mydomain.POD,mydomain.DMD,axis=0,timesteps=300)
show(out)

#Print timing
print ("Initialization: "+mydomain.wall
       +"\nPOD: "+mydomain.POD.wall
       +"\nDMD: "+mydomain.DMD.wall)

```

