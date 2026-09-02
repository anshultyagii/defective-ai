import scipy.io
import sys
mat = scipy.io.loadmat(sys.argv[1])
meas = mat['measurement'][0, 0]
for name in meas.dtype.names:
    print(f"{name} shape: {meas[name].shape}")
