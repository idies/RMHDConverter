import numpy as np
import sys
import os
sys.path.append('/home/clalescu/repos/personal/py/lib')
import chichi_misc as cm

N = 2048
n = N // 8

nz = (512 // 8)**3

iteration = 144000
iter0 = 138000

rdata = {}

field_name = ['ux', 'uy', 'uz', 'bx', 'by', 'bz']

for k in [2, 3, 5, 6]:
    fname = os.path.join(
        '/datascope/tdbrmhd04/real_space/b2048n/',
        'R{0:0>6}QNP.{1}'.format(iteration, k))
    rdata[field_name[k-1]] = np.memmap(
            fname,
            dtype = '<f4',
            mode = 'r',
            shape = (N, N, N))

zdir = np.array(range(0, n**3, n**3 // 4))
zfolder = ['/datascope/tdbrmhd05/',
           '/datascope/tdbrmhd06/',
           '/datascope/tdbrmhd07/',
           '/datascope/tdbrmhd08/']

for counter in range(10):
    z = np.random.randint(n**3 // 64)
    i = cm.zindex_to_grid3D(z)
    print(z)

    zz = z % nz
    zfile = z - zz

    # velocity field
    zdata = np.memmap(
            os.path.join(
                zfolder[np.searchsorted(zdir, z+1)-1],
                'RMHD_u_t{0:0>4x}_z{1:0>7x}'.format(iteration - iter0, zfile)),
            dtype = '<f4',
            mode = 'r',
            shape = (nz, 8, 8, 8, 2))
    zcubbie = zdata[zz].copy()
    err_uy = np.max(
        np.abs(zcubbie[..., 0] -
               rdata['uy'][i[0]*8:(i[0]+1)*8,
                           i[1]*8:(i[1]+1)*8,
                           i[2]*8:(i[2]+1)*8]))
    err_uz = np.max(
        np.abs(zcubbie[..., 1] -
               rdata['uz'][i[0]*8:(i[0]+1)*8,
                           i[1]*8:(i[1]+1)*8,
                           i[2]*8:(i[2]+1)*8]))
    print(rdata['uy'][i[0]*8 + 3, i[1]*8 + 2, i[2]*8 + 5],
          rdata['uz'][i[0]*8 + 3, i[1]*8 + 2, i[2]*8 + 5],
          zcubbie[3, 2, 5])
    # magnetic field
    zdata = np.memmap(
            os.path.join(
                zfolder[np.searchsorted(zdir, z+1)-1],
                'RMHD_b_t{0:0>4x}_z{1:0>7x}'.format(iteration - iter0, zfile)),
            dtype = '<f4',
            mode = 'r',
            shape = (nz, 8, 8, 8, 2))
    zcubbie = zdata[zz].copy()
    err_by = np.max(
        np.abs(zcubbie[..., 0] -
               rdata['by'][i[0]*8:(i[0]+1)*8,
                           i[1]*8:(i[1]+1)*8,
                           i[2]*8:(i[2]+1)*8]))
    err_bz = np.max(
        np.abs(zcubbie[..., 1] -
               rdata['bz'][i[0]*8:(i[0]+1)*8,
                           i[1]*8:(i[1]+1)*8,
                           i[2]*8:(i[2]+1)*8]))
    print(rdata['by'][i[0]*8 + 3, i[1]*8 + 2, i[2]*8 + 5],
          rdata['bz'][i[0]*8 + 3, i[1]*8 + 2, i[2]*8 + 5],
          zcubbie[3, 2, 5])
    print(err_uy, err_uz, err_by, err_bz)

