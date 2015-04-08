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
import pyfftw
import os
import ctypes

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
    for k in range(a.shape[0]//8):
        for j in range(a.shape[1]//8):
            for i in range(a.shape[2]//8):
                z = cm.grid3D_to_zindex(np.array([k, j, i]))
                c[z] = a[8*k:8*(k+1), 8*j:8*(j+1), 8*i:8*(i+1)]
    return c

class zshuffler:
    def __init__(self, threads = 8):
        self.lib = np.ctypeslib.load_library(
            'libzshuffle',
            os.path.abspath('./'))
        self.threads = threads
        return None
    def __call__(
            self,
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
        self.lib.shuffle_threads(
            ctypes.c_int(a.shape[0]),
            ctypes.c_int(a.shape[3]),
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.threads))
        return c

def main(
        opt):
    a = np.random.random((opt.N, opt.N, opt.N, 2)).astype(np.float32)
    b = array_to_8cubes(a)
    z = zshuffler(opt.threads)
    c = z(a)
    print(np.max(np.abs(b - c)))
    return None

import cProfile
import pstats
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-N',
        type = int,
        dest = 'N',
        metavar = 'N',
        default = 256)
    parser.add_argument(
        '-t',
        type = int,
        dest = 'threads',
        metavar = 'threads',
        default = 8)
    opt = parser.parse_args()

    cProfile.run('main(opt)', 'profile')
    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(10)

