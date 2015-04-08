#! /usr/bin/env python3.4

import multiprocessing as mp

from file_mover import FileMover
from file_converter import FileConverter

def main(opt):
    src_dir = '/datascope/tdbrmhd01/o2048n/'
    first_iter = 138000 + opt.start
    if first_iter > 144904:
        src_dir = '/datascope/tdbrmhd03/o2048n/'
    c = FileConverter(
        n = 1364,
        N = 2048,
        nfiles = 64,
        fft_threads = 32,
        out_threads = 8,
        iter0 = 138000,
        src_dir = src_dir,
        dst_dir = '/export/scratch0/RMHD/zindex/',
        job_done_pattern = 'queue_files/converted_t{0}',
        space_needed = (2*4*(2048**3))*2*3,
        announce_folder = 'queue_files')
    ofile_list = c.get_ofile_list(field = 'u')
    nfiles = len(ofile_list)
    out_folder = ['/datascope/tdbrmhd05/',
                  '/datascope/tdbrmhd06/',
                  '/datascope/tdbrmhd07/',
                  '/datascope/tdbrmhd08/']
    ndirs = len(out_folder)
    assert(nfiles % ndirs == 0)
    um = [FileMover(
             src_dir = '/export/scratch0/RMHD/zindex',
             dst_dir = out_folder[i],
             src_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             dst_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             ready_to_start_pattern = 'queue_files/uconverted_t{0}',
             job_done_pattern = 'queue_files/movedu_t{0}')
         for i in range(ndirs)]
    ofile_list = c.get_ofile_list(field = 'b')
    bm = [FileMover(
             src_dir = '/export/scratch0/RMHD/zindex',
             dst_dir = out_folder[i],
             src_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             dst_pattern = ofile_list[i*nfiles//ndirs:(i+1)*nfiles//ndirs],
             ready_to_start_pattern = 'queue_files/bconverted_t{0}',
             job_done_pattern = 'queue_files/movedb_t{0}')
         for i in range(ndirs)]
    iterations = range(first_iter, first_iter + 4*opt.niter, 4)
    pc = mp.Process(
             target = c.work,
             args = (iterations,))
    pm = [mp.Process(
              target = mm.work,
              args = ([i - 138000 for i in iterations],))
          for mm in um + bm]
    for p in pm + [pc]:
        p.start()
    for p in pm + [pc]:
        p.join()
    return None

import cProfile
import pstats
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--iteration',
        type = int,
        dest = 'start',
        help = 'start iteration',
        metavar = 'T',
        default = 144000 - 138000)
    parser.add_argument(
        '-n', '--niter',
        type = int,
        dest = 'niter',
        help = 'number of iterations to process',
        metavar = 'N',
        default = 1)
    opt = parser.parse_args()

    cProfile.run('main(opt)', 'profile')
    p = pstats.Stats('profile')
    p.sort_stats('cumulative').print_stats(10)
    p.sort_stats('time').print_stats(10)

