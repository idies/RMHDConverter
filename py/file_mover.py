import os
import subprocess

from file_worker import FileWorkerBase

class FileMover(FileWorkerBase):
    def __init__(
            self,
            dst_dir = None,
            ready_to_start_pattern = None,
            job_done_pattern = None,
            space_needed = 0,
            src_dir = None,
            src_pattern = None,
            dst_pattern = None):
        super(FileMover, self).__init__(
            ready_to_start_pattern = ready_to_start_pattern,
            dst_dir = dst_dir,
            job_done_pattern = job_done_pattern,
            space_needed = space_needed)
        self.src_dir = src_dir
        self.src_pattern = src_pattern
        self.dst_pattern = dst_pattern
        return None
    def iterate_one_file(
            self,
            ipattern,
            opattern,
            iteration):
        src_file = os.path.abspath(os.path.join(
            self.src_dir,
            ipattern.format(iteration)))
        # first case: we're moving things
        if type(self.dst_dir) != type(None):
            if not os.path.exists(self.dst_dir):
                os.mkdir(self.dst_dir)
            dst_file = os.path.abspath(os.path.join(
                self.dst_dir,
                opattern.format(iteration)))
            #subprocess.check_call(['rsync', '-av', src_file, dst_file])
            subprocess.check_call(['cp', src_file, dst_file])
        # now we're deleting things, whether they're copied or not
        subprocess.check_call(['rm', src_file])
        return None
    def iterate(self, iteration):
        if type(self.src_pattern) == type('string'):
            self.iterate_one_file(
                self.src_pattern, self.dst_pattern, iteration)
        elif type(self.src_pattern) == type([]):
            for f in range(len(self.src_pattern)):
                self.iterate_one_file(
                    self.src_pattern[f],
                    self.dst_pattern[f],
                    iteration)
        return None

