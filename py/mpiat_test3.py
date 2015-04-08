
# coding: utf-8

# In[1]:

import numpy as np
import subprocess
get_ipython().magic('matplotlib nbagg')
import matplotlib.pyplot as plt
import pyfftw


# In[10]:

def generate_data_3D(
        n,
        dtype = np.complex128,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n % 2 == 0)
    a = np.zeros((n, n, n/2+1), dtype = dtype)
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[-n/2+1:n/2+1, -n/2+1:n/2+1, 0:n/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n//2+1, axis = 0)
    k = np.roll(k, n//2+1, axis = 1)
    a /= k**p
    a[0, :, :] = 0
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    ii = np.where(k == 0)
    a[ii] = 0
    ii = np.where(k > n/3)
    a[ii] = 0
    return a

n = 31*8
N = 512

Kdata0 = generate_data_3D(n, p = 2).astype(np.complex64)
Kdata1 = generate_data_3D(n, p = 2).astype(np.complex64)
Kdata2 = generate_data_3D(n, p = 2).astype(np.complex64)
Kdata0.T.copy().astype('>c8').tofile("Kdata0")
Kdata1.T.copy().astype('>c8').tofile("Kdata1")
Kdata2.T.copy().astype('>c8').tofile("Kdata2")


# In[11]:

def padd_with_zeros(
        a,
        n,
        odtype = None):
    if (type(odtype) == type(None)):
        odtype = a.dtype
    assert(a.shape[0] <= n)
    b = np.zeros((n, n, n/2 + 1), dtype = odtype)
    m = a.shape[0]
    b[     :m/2,      :m/2, :m/2+1] = a[     :m/2,      :m/2, :m/2+1]
    b[     :m/2, n-m/2:   , :m/2+1] = a[     :m/2, m-m/2:   , :m/2+1]
    b[n-m/2:   ,      :m/2, :m/2+1] = a[m-m/2:   ,      :m/2, :m/2+1]
    b[n-m/2:   , n-m/2:   , :m/2+1] = a[m-m/2:   , m-m/2:   , :m/2+1]
    return b

def transform_py(bla):
    b = padd_with_zeros(bla, N)
    c = np.zeros((N, N, N), np.float32)
    t = pyfftw.FFTW(
        b, c,
        axes = (0, 1, 2),
        direction = 'FFTW_BACKWARD',
        flags = ('FFTW_ESTIMATE',),
        threads = 2)
    t.execute()
    return c

import chichi_misc as cm

def array_to_8cubes(
    a,
    odtype = None):
    assert(len(a.shape) >= 3)
    assert((a.shape[0] % 8 == 0) and
           (a.shape[1] % 8 == 0) and
           (a.shape[2] % 8 == 0))
    if type(odtype) == type(None):
        odtype = a.dtype
    c = np.zeros(
        ((((a.shape[0] // 8)*(a.shape[1] // 8)*(a.shape[2] // 8)),) +
         (8, 8, 8) +
         a.shape[3:]),
        dtype = odtype)
    zi = np.zeros( c.shape[0], dtype = np.int)
    ri = np.zeros((c.shape[0], 3, 2), dtype = np.int)
    ii = 0
    for k in range(a.shape[0]//8):
        for j in range(a.shape[1]//8):
            for i in range(a.shape[2]//8):
                tindex = np.array([k, j, i])
                zi[ii] = cm.grid3D_to_zindex(tindex)
                ri[ii, 0] = np.array([8*tindex[0], 8*(tindex[0]+1)])
                ri[ii, 1] = np.array([8*tindex[1], 8*(tindex[1]+1)])
                ri[ii, 2] = np.array([8*tindex[2], 8*(tindex[2]+1)])
                ii += 1
    for ii in range(zi.shape[0]):
        c[zi[ii]] = a[ri[ii, 0, 0]:ri[ii, 0, 1],
                      ri[ii, 1, 0]:ri[ii, 1, 1],
                      ri[ii, 2, 0]:ri[ii, 2, 1]]
    return c

d0 = transform_py(Kdata0)
d1 = transform_py(Kdata1)
d2 = transform_py(Kdata2)

Rdata_py_tmp = np.array([d0, d1, d2]).transpose((1, 2, 3, 0))

Rdata_py = array_to_8cubes(Rdata_py_tmp)

# i0 = np.random.randint(16)
# i1 = np.random.randint(16)
# i2 = np.random.randint(16)
# z = cm.grid3D_to_zindex(np.array([i0, i1, i2]))


# In[12]:

def compute_cpp_data(
        branch = None,
        nfiles = 16):
    if not (type(branch) == type(None)):
        subprocess.call(['git', 'checkout', branch])
    if subprocess.call(['make', 'full.elf']) == 0:
        subprocess.call([#'valgrind',
                         #'--tool=callgrind',
                         #'--callgrind-out-file=tmp.txt',
                         'time',
                         'mpirun.mpich',
                         '-np',
                         '8',
                         './full.elf',
                         '{0}'.format(n),
                         '{0}'.format(N),
                         '{0}'.format(nfiles),
                         '3'])
    else:
        print ('compilation error')
        return None
    
def get_cpp_data(
        branch = None,
        run = True,
        nfiles = 16):
    if run:
        for nf in range(nfiles):
            subprocess.call(
                ['rm',
                 'Rdata_z{0:0>7x}'.format(nf*Rdata_py.shape[0]//nfiles)])
        compute_cpp_data(branch, nfiles = nfiles)
    Rdata = []
    for nf in range(nfiles):
        Rdata.append(np.fromfile(
        'Rdata_z{0:0>7x}'.format(nf*Rdata_py.shape[0]//nfiles),
        dtype = np.float32).reshape(-1, 8, 8, 8, 3))
    return np.concatenate(Rdata)

#Rdata = get_cpp_data(branch = 'develop')
# develop says 30 secs, inplace fft is 28 secs
#Rdata = get_cpp_data(branch = 'feature-inplace_fft')
Rdata = get_cpp_data(run = True, nfiles = 16)


# In[13]:

distance = np.max(np.abs(Rdata_py - Rdata), axis = (1, 2, 3, 4))
print(np.max(distance))
if np.max(distance) > 1e-5:
    ax = plt.figure(figsize=(6,2)).add_subplot(111)
    ax.plot(distance)
    i0 = np.random.randint(8)
    i1 = np.random.randint(8)
    i2 = np.random.randint(8)
    z = cm.grid3D_to_zindex(np.array([i0, i1, i2]))
    #z = 0
    print(cm.zindex_to_grid3D(z))
    s = np.max(np.abs(Rdata_py[None, z, :, :, :, 1] - Rdata[..., 1]),
               axis = (1, 2, 3))
    z1 = np.argmin(s)
    print(z, z1, s[z1])
        #print(Rdata[z1] - Rdata_py[z1])
    ta0 = Rdata_py.ravel()
    ta1 = Rdata.ravel()
    print (Rdata_py[254:259, 7, 4, 3, 1])
    print (Rdata[254:259, 7, 4, 3, 1])
    print (ta0[ta0.shape[0]/2-1:ta0.shape[0]/2+7])
    print (ta1[ta1.shape[0]/2-1:ta1.shape[0]/2+7])
else:
    print('distance is small')
print(np.max(np.abs(Rdata)))


# In[ ]:



