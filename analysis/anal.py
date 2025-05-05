from uncertainties import ufloat
from uncertainties import unumpy
import numpy as np
uarray = unumpy.uarray


eta = ufloat(0.92, 0.286+0.04)
vbig = uarray([0.024,  0.062], [0.002, 0.002])
vsm = uarray([ 0.015, 0.062], [0.002, 0.002])

Rbig = 2.06e-6/2
Rsml = 1.5e-6/2

Fbig = 6*np.pi*eta*vbig*Rbig*1e-6
Fsml = 6*np.pi*eta*vsm*Rsml*1e-6

print(f"{Fbig}")
print(f"{Fsml}")
print(f"{Fbig/(Rbig*1e9/10)}")
print(f"{Fsml/(Rsml*1e9/10)}")
