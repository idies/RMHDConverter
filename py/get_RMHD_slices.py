import numpy as np
import sys
import os
import matplotlib.pyplot as plt

sys.path.append('/home/clalescu/repos/personal/py/lib')
import chichi_misc as cm

from RMHD_converter import parmap

class frame_viewer:
    def __init__(self, iteration = 0):
        self.N = 2048
        self.n = self.N//8
        self.nz = (512 // 8)**3
        self.iteration = iteration
        self.zdir = np.array(range(0, self.n**3, self.n**3 // 4))
        self.zfolder = ['/datascope/tdbrmhd05/',
                        '/datascope/tdbrmhd06/',
                        '/datascope/tdbrmhd07/',
                        '/datascope/tdbrmhd08/']
        self.zfile = np.array(range(0, self.n**3, self.n**3 // 64))
        self.datafile = {'u' : [],
                         'b' : []}
        for z in self.zfile:
            for key in ['u', 'b']:
                self.datafile[key].append(np.memmap(
                    os.path.join(
                        self.zfolder[np.searchsorted(self.zdir, z+1)-1],
                        ('RMHD_' +
                         key +
                         '_t{0:0>4x}_z{1:0>7x}'.format(self.iteration, z))),
                    dtype = '<f4',
                    mode = 'r',
                    shape = (self.nz, 8, 8, 8, 2)))
        return None
    def __call__(self, key, k, j, i):
        assert(key in ['u', 'b'])
        assert(type(k) == type(np.zeros(1, dtype = np.int)))
        assert(type(j) == type(np.zeros(1, dtype = np.int)))
        assert(type(i) == type(np.zeros(1, dtype = np.int)))
        assert(k.shape == j.shape == i.shape)
        z = cm.grid3D_to_zindex(np.array([k//8, j//8, i//8]))
        result = np.zeros(k.shape + (2,), dtype = np.float32)
        zfi = np.searchsorted(self.zfile, z+1)-1
        for ii in np.ndindex(*k.shape):
            result[ii] = self.datafile[key][zfi[ii]][z[ii]-zfi[ii]*self.nz, k[ii]%8, j[ii]%8, i[ii]%8]
        return result

def main(rseed = 1):
    i0, i1 = np.mgrid[0:2048, 0:2048]
    i2 = np.zeros(i0.shape, i0.dtype)
    rseed = 2
    np.random.seed(rseed)
    n2 = np.random.randint(2048, size = 3)

    bname = '2D_slices/data_rs{0}'.format(rseed)

    def get_slices(iteration):
        iname = '{0:0>4x}'.format(iteration)
        print('at iteration ' + iname)
        fname = bname + '_u_t' + iname + '.npy'
        if not os.path.exists(fname):
            print('computing for iteration ' + iname)
            a = frame_viewer(iteration = iteration)
            d = np.array([
                a('u', i0, i1, i2 + n2[0]),
                a('u', i1, i2 + n2[1], i0),
                a('u', i2 + n2[2], i0, i1)])
            np.save(bname + '_u_t' + iname, d)
            d = np.array([
                a('b', i0, i1, i2 + n2[0]),
                a('b', i1, i2 + n2[1], i0),
                a('b', i2 + n2[2], i0, i1)])
            np.save(bname + '_b_t' + iname, d)
        return None

    parmap(get_slices, range(0, 6*2**10, 2**7), nprocs = 32)
    return None

import cProfile
import pstats

if __name__ == '__main__':
    cProfile.run('main()', 'profile')
    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('time').print_stats(10)

