import numpy as np
import os
import ctypes
import subprocess
import time

from file_mover import FileWorkerBase
from RMHD_converter import converter
from RMHD_converter import parmap

class FileConverter(FileWorkerBase):
    def __init__(
            self,
            dst_dir = None,
            ready_to_start_pattern = None,
            job_done_pattern = None,
            announce_folder = None,
            space_needed = 0,
            src_dir = None,
            n = None,
            N = None,
            iter0 = 138000,
            nfiles = 64,
            fft_threads = 32,
            out_threads = 2):
        super(FileConverter, self).__init__(
            ready_to_start_pattern = ready_to_start_pattern,
            dst_dir = dst_dir,
            job_done_pattern = job_done_pattern,
            space_needed = space_needed)
        self.src_dir = src_dir
        self.announce_folder = announce_folder
        self.converter = converter(
            dst_dir = self.dst_dir,
            src_dir = self.src_dir,
            iter0 = iter0,
            n = n,
            N = N,
            nfiles = nfiles,
            fft_threads = fft_threads,
            out_threads = out_threads)
        self.iter0 = iter0
        return None
    def iterate(self, iteration):
        self.converter.transform(
            iteration = iteration)
        return None
    def get_ofile_list(
            self,
            field = 'u'):
        ofile_list = []
        for z in range(
                0,
                self.converter.rzdata.shape[0],
                self.converter.cubbies_per_file):
            ofile_list.append(
                self.converter.dst_format.replace(
                    '{1:0>4x}', '{1}').format(field, '{0:0>4x}', z))
        return ofile_list
    def work(self, iter_list):
        for i in iter_list:
            self.start(i - self.iter0)
            for info in [('u', 2, 3),
                         ('b', 5, 6)]:
                print('currently working on ' +
                      info[0] +
                      ' for iteration {0} = {1:0>4x}'.format(i, i - self.iter0))
                self.converter.shuffle_lib.array3D_to_zero_threads(
                    ctypes.c_int(self.converter.kdata.shape[0]),
                    ctypes.c_int(self.converter.kdata.shape[1]),
                    ctypes.c_int(self.converter.kdata.shape[2]),
                    ctypes.c_int(self.converter.kdata.shape[3]*2),
                    self.converter.kdata.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_int(self.converter.fft_threads))
                if not type(self.announce_folder) == type(None):
                    while os.path.exists(
                        os.path.join(self.announce_folder, 'reading')):
                        time.sleep(.2)
                    subprocess.call(['touch',
                                     os.path.join(self.announce_folder, 'reading')])
                print('reading y component of field')
                self.converter.read_data(finfo = (i, info[1], 0))
                print('reading z component of field')
                self.converter.read_data(finfo = (i, info[2], 1))
                if not type(self.announce_folder) == type(None):
                    subprocess.call(['rm',
                                     os.path.join(self.announce_folder, 'reading')])
                print('executing fftw plan')
                self.converter.plan.execute()
                print('performing z-order shuffle')
                self.converter.zshuffle()
                write_info = [(info[0], i - self.iter0, z)
                              for z in range(
                                  0,
                                  self.converter.rzdata.shape[0],
                                  self.converter.cubbies_per_file)]
                if type(self.converter.dst_dir) == type([]):
                    shuffled_write_info = []
                    for i in range(len(self.converter.dst_dir)):
                        shuffled_write_info += \
                            write_info[i::len(self.converter.dst_dir)]
                else:
                    shuffled_write_info = write_info
                print('now writing files')
                parmap(self.converter.write_data,
                       shuffled_write_info,
                       nprocs = self.converter.out_threads)
                if not type(self.announce_folder) == type(None):
                    subprocess.call(
                        ['touch',
                         os.path.join(
                             self.announce_folder,
                             info[0] + 'converted_t{0}'.format(i - self.iter0))])
            self.finish(i - self.iter0)
        return None

