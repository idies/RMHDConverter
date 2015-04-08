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
import pickle
import gzip
import ctypes
import multiprocessing as mp
import multiprocessing.sharedctypes as sct
from multiprocessing.pool import ThreadPool
import subprocess

from zindex import zindex_to_grid3D, grid3D_to_zindex

#######################################################################
# got the two mp things from
# https://stackoverflow.com/a/16071616/4205267
def mpfun(f,q_in,q_out):
    while True:
        i,x = q_in.get()
        if i is None:
            break
        q_out.put((i,f(x)))

def parmap(f, X, nprocs = mp.cpu_count()):
    q_in   = mp.Queue(1)
    q_out  = mp.Queue()

    proc = [mp.Process(target=mpfun,args=(f,q_in,q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]
#######################################################################

#######################################################################
class converter:
    def __init__(self,
                 n = 1364,
                 N = 2048,
                 iter0 = 138000,
                 nfiles = 64,
                 fft_threads = 32,
                 out_threads = 4,
                 src_dir = './',
                 dst_dir = './',
                 src_format = 'K{0:0>6}QNP{1:0>3}',
                 dst_format = 'RMHD_{0}_t{1:0>4x}_z{2:0>7x}'):
        self.src_format = src_format
        self.dst_format = dst_format
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.n = n
        self.N = N
        self.iter0 = iter0
        self.nfiles = nfiles
        self.kdata = pyfftw.n_byte_align_empty(
                (self.N//2+1, 2, self.N, self.N),
                pyfftw.simd_alignment,
                dtype = np.complex64)
        self.rrdata = pyfftw.n_byte_align_empty(
                (self.N, self.N, self.N, 2),
                pyfftw.simd_alignment,
                dtype = np.float32)
        self.rzdata = pyfftw.n_byte_align_empty(
                ((self.N//8) * (self.N//8) * (self.N//8),
                 8*8*8*2),
                pyfftw.simd_alignment,
                dtype = np.float32)
        if type(self.dst_dir) == type([]):
            self.zdir = np.array(
                range(0,
                      self.rzdata.shape[0],
                      self.rzdata.shape[0] // len(self.dst_dir)))
        self.cubbies_per_file = self.rzdata.shape[0] // self.nfiles
        if (os.path.isfile('fftw_wisdom.pickle.gz')):
            pyfftw.import_wisdom(
                pickle.load(gzip.open('fftw_wisdom.pickle.gz', 'rb')))
        print('about to initialize the fftw plan, which can take a while')
        self.plan = pyfftw.FFTW(
                self.kdata.transpose(3, 2, 0, 1), self.rrdata,
                axes = (0, 1, 2),
                direction = 'FFTW_BACKWARD',
                flags = ('FFTW_MEASURE',
                         'FFTW_DESTROY_INPUT'),
                threads = fft_threads)
        print('finalized fftw initialization')
        bla = pyfftw.export_wisdom()
        pickle.dump(bla, gzip.open('fftw_wisdom.pickle.gz', 'wb'))
        self.fft_threads = fft_threads
        self.out_threads = out_threads
        self.shuffle_lib = np.ctypeslib.load_library(
            'libzshuffle.so',
            os.path.abspath(os.path.join(
                os.path.expanduser('~'), 'repos/RMHD_converter/C-shuffle')))
        return None
    def read_data(self, finfo = (138000, 2, 0)):
        fname = os.path.join(
            self.src_dir,
            self.src_format.format(finfo[0], finfo[1]))
        file_data = np.fromfile(fname, dtype = '>c8').reshape(
            self.n//2+1, self.n, self.n)
        self.kdata[:self.n//2+1,
                   finfo[2],
                   :self.n//2,
                   :self.n//2] = file_data[:self.n//2+1,
                                           :self.n//2,
                                           :self.n//2]
        self.kdata[:self.n//2+1,
                   finfo[2],
                   self.N-self.n//2:,
                   :self.n//2] = file_data[:self.n//2+1,
                                           self.n-self.n//2:
                                           self.n,
                                           :self.n//2]
        self.kdata[:self.n//2+1,
                   finfo[2],
                   :self.n//2,
                   self.N-self.n//2:] = file_data[:self.n//2+1,
                                                  :self.n//2,
                                                  self.n-self.n//2:]
        self.kdata[:self.n//2+1,
                   finfo[2],
                   self.N-self.n//2:,
                   self.N-self.n//2:] = file_data[:self.n//2+1,
                                                  self.n-self.n//2:self.n,
                                                  self.n-self.n//2:]
        return None
    def write_data(self, finfo = ('u', 0, 0)):
        if type(self.dst_dir) == type([]):
            fname = os.path.join(
                self.dst_dir[np.searchsorted(self.zdir, finfo[2]+1)-1],
                self.dst_format.format(finfo[0], finfo[1], finfo[2]))
        else:
            fname = os.path.join(
                self.dst_dir,
                self.dst_format.format(finfo[0], finfo[1], finfo[2]))
        print('writing ' + fname)
        self.rzdata[finfo[2]:finfo[2]+self.cubbies_per_file].tofile(fname)
    def zshuffle(self):
        self.shuffle_lib.shuffle_threads(
            ctypes.c_int(self.N),
            ctypes.c_int(2),
            self.rrdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.rzdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.fft_threads//2))
        return None
    def transform(self, iteration = 138000):
        for info in [('u', 2, 3),
                     ('b', 5, 6)]:
            print('currently working on ' +
                  info[0] +
                  ' for iteration {0} = {1:0>4x}'.format(iteration, iteration - self.iter0))
            self.shuffle_lib.array3D_to_zero_threads(
                ctypes.c_int(self.kdata.shape[0]),
                ctypes.c_int(self.kdata.shape[1]),
                ctypes.c_int(self.kdata.shape[2]),
                ctypes.c_int(self.kdata.shape[3]*2),
                self.kdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(self.fft_threads))
            print('reading y component of field')
            self.read_data(finfo = (iteration, info[1], 0))
            print('reading z component of field')
            self.read_data(finfo = (iteration, info[2], 1))
            print('executing fftw plan')
            self.plan.execute()
            print('performing z-order shuffle')
            self.zshuffle()
            write_info = [(info[0], iteration - self.iter0, z)
                          for z in range(0, self.rzdata.shape[0], self.cubbies_per_file)]
            if type(self.dst_dir) == type([]):
                shuffled_write_info = []
                for i in range(len(self.dst_dir)):
                    shuffled_write_info += write_info[i::len(self.dst_dir)]
            else:
                shuffled_write_info = write_info
            print('now writing files')
            parmap(self.write_data,
                   shuffled_write_info,
                   nprocs = self.out_threads)
        return None

def main(opt):
    bla = converter(
        dst_dir = ['/datascope/tdbrmhd05/',
                   '/datascope/tdbrmhd06/',
                   '/datascope/tdbrmhd07/',
                   '/datascope/tdbrmhd08/'],
        src_dir = '/datascope/tdbrmhd01/o2048n/',
        nfiles = 64,
        n = 1364,
        N = 2048,
        fft_threads = 32,
        out_threads = 8)
    for i in range(opt.iteration, opt.iteration + 4*opt.niter, 4):
        bla.transform(iteration = i)
    return None

import cProfile
import pstats
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--iteration',
        type = int,
        dest = 'iteration',
        help = 'iteration',
        metavar = 'T',
        default = 144000)
    parser.add_argument(
        '-n', '--no-of-iterations',
        type = int,
        dest = 'niter',
        help = 'number of iterations to process',
        metavar = 'N',
        default = 1)
    opt = parser.parse_args()

    cProfile.run('main(opt)', 'profile')
    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(15)
    p.sort_stats('time').print_stats(15)

