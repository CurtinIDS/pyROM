# pyROM - Reduced Order Modelling Framework for Python

## Background

Solving a PDE numerically for mesh based/time dependent codes usually involves iterating through the whole spatial basis that represents the underlying system. Reduced Order Modelling involves scanning for dominant "modes" that contain most information of the happening sections. Quantitatively, a 10^4x10^4 matrix could be transformed in to a 100x3 matrix, evolution of which and retransformation back to 10^4x10^4, may result in negligible error.

## Algorithm

1.  Fetch stiffness matrix, snapshot matrix(evolved for Δt timesteps).
2.  Compute PCA/DMD/POD (user choice) to select the first N modes. Ex. - [Φ_n]=[Φ_1,Φ_23,Φ_10]
3.  Solve σ⍺=∂⍺/∂t, where ⍺ is a kxk matrix(k is min. dimension of [Φ_n], the dependent variable C=[Φ_n][⍺].
4.  Transform [Φ_n] back to C after t timesteps
*Fetch snapshots as and when needed
