#! /usr/bin/env python3.4

import os
import subprocess
import multiprocessing as mp
import time

from file_mover import FileMover
from file_generator import FileGenerator
from file_converter import FileConverter

def main():
    g = FileGenerator(
        n = 31*4,
        N = 256,
        dst_dir = 'src_folder',
        dst_pattern = ['K{0:0>6}QNP002',
                       'K{0:0>6}QNP003',
                       'K{0:0>6}QNP005',
                       'K{0:0>6}QNP006'],
        job_done_pattern = 'queue_files/created_t{0}')
    c = FileConverter(
        n = 31*4,
        N = 256,
        nfiles = 16,
        fft_threads = 8,
        out_threads = 1,
        iter0 = 0,
        src_dir = 'src_folder',
        dst_dir = 'scratch_folder',
        ready_to_start_pattern = 'queue_files/created_t{0}',
        job_done_pattern = 'queue_files/converted_t{0}',
        space_needed = 1024**3,
        announce_folder = 'queue_files')
    ofile_list = c.get_ofile_list(field = 'u')
    nfiles = len(ofile_list)
    out_folder = ['out05',
                  'out06',
                  'out07',
                  'out08']
    ndirs = len(out_folder)
    assert(nfiles % ndirs == 0)
    um = [FileMover(
             src_dir = 'scratch_folder',
             dst_dir = out_folder[i],
             src_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             dst_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             ready_to_start_pattern = 'queue_files/uconverted_t{0}',
             job_done_pattern = 'queue_files/umoved_t{0}')
         for i in range(ndirs)]
    ofile_list = c.get_ofile_list(field = 'b')
    bm = [FileMover(
             src_dir = 'scratch_folder',
             dst_dir = out_folder[i],
             src_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             dst_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             ready_to_start_pattern = 'queue_files/bconverted_t{0}',
             job_done_pattern = 'queue_files/bmoved_t{0}')
         for i in range(ndirs)]
    pg = mp.Process(target = g.work, args = (range(6),))
    pc = mp.Process(target = c.work, args = (range(6),))
    pm = [mp.Process(target = mm.work, args = (range(6),))
          for mm in um+bm]
    for p in pm + [pg, pc]:
        p.start()
    for p in pm + [pg, pc]:
        p.join()
    return None

if __name__ == '__main__':
    main()

