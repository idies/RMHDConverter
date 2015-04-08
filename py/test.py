#! /usr/bin/env python3.4

########################################################################
#
#  Copyright 2015 Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Contact: turbulence@pha.jhu.edu
# Website: http://turbulence.pha.jhu.edu/
#
########################################################################



import numpy as np
import cProfile
import pstats
import pyfftw
import chichi_misc as cm

from RMHD_converter import converter
from file_generator import generate_data_3D, padd_with_zeros

def transform_py(bla, N):
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
    for k in range(a.shape[0]//8):
        for j in range(a.shape[1]//8):
            for i in range(a.shape[2]//8):
                tindex = np.array([k, j, i])
                z = cm.grid3D_to_zindex(tindex)
                c[z] = a[8*k:8*(k+1),
                         8*j:8*(j+1),
                         8*i:8*(i+1)]
    return c

def check_distance(
        a, b,
        label = 'u'):
    distance = np.max(np.abs(a - b), axis = (1, 2, 3, 4))
    if np.max(distance) > 1e-5:
        print('distance for ' + label + ' is {0} (BIG)'.format(np.max(distance)))
        print('distances for first cubbie are\n{0}\n{1}\n{2}'.format(
            np.max(np.abs(a[0] - b[0])),
            np.max(np.abs(a[0, 0] - b[0, 0])),
            np.abs(a[0, 0, 0] - b[0, 0, 0])))
        print('values for first cubbie line are\n{0}\n{1}'.format(
            a[0, 0, 0], b[0, 0, 0]))
    else:
        print('distance for ' + label + ' is {0}'.format(np.max(distance)))
    return None

def main_small():
    n = 31*4
    N = 256
    nfiles = 4

    Kdata = {}
    d = {}
    for k in ['2', '3', '5', '6']:
        Kdata[k] = generate_data_3D(n, p = 2).astype(np.complex64)
        Kdata[k].T.copy().astype('>c8').tofile("Kdata" + k)
        d[k] = transform_py(Kdata[k], N)
    Rdatau0 = array_to_8cubes(
        np.array([d['2'], d['3']]).transpose((1, 2, 3, 0)).copy())
    Rdatab0 = array_to_8cubes(
        np.array([d['5'], d['6']]).transpose((1, 2, 3, 0)).copy())
    c = converter(
        n = n,
        N = N,
        iter0 = 0,
        nfiles = nfiles,
        fft_threads = 8,
        src_format = 'Kdata{1}',
        dst_format = 'Rdata_{0}_z{2:0>7x}',
        dst_dir = ['test0', 'test1'])
    c.transform(iteration = 0)
    Rdata1 = []
    for nf in range(nfiles):
        Rdata1.append(np.fromfile(
        'Rdata_u_z{0:0>7x}'.format(nf*Rdatau0.shape[0]//nfiles),
        dtype = np.float32).reshape(-1, 8, 8, 8, 2))
    Rdatau1 = np.concatenate(Rdata1)
    check_distance(Rdatau0, Rdatau1, 'u')
    Rdata1 = []
    for nf in range(nfiles):
        Rdata1.append(np.fromfile(
        'Rdata_b_z{0:0>7x}'.format(nf*Rdatau0.shape[0]//nfiles),
        dtype = np.float32).reshape(-1, 8, 8, 8, 2))
    Rdatab1 = np.concatenate(Rdata1)
    check_distance(Rdatab0, Rdatab1, 'b')
    return None

def main_big():
    c = converter(
        src_dir = '/datascope/tdbrmhd01/o2048n/',
        dst_dir = 'zindex/')
    c.transform(iteration = 144000)
    return None

if __name__ == '__main__':
    cProfile.run('main_small()', 'profile')
    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(15)
    p.sort_stats('time').print_stats(15)

