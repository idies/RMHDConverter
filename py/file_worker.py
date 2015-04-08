import os
import subprocess
import time

class FileWorkerBase(object):
    def __init__(
            self,
            dst_dir = None,
            ready_to_start_pattern = None,
            job_done_pattern = None,
            space_needed = 0):
        self.dst_dir = dst_dir
        self.ready_to_start_pattern = ready_to_start_pattern
        self.job_done_pattern = job_done_pattern
        self.space_needed = space_needed
        return None
    def get_free_space(self):
        if type(self.dst_dir) == type(None):
            return None
        else:
            statvfs = os.statvfs(self.dst_dir)
            return statvfs.f_frsize * statvfs.f_bavail
    def finish(self, iteration):
        subprocess.call(['touch', self.job_done_pattern.format(iteration)])
        return None
    def start(self, iteration):
        # do we have anything to wait for?
        if not (type(self.ready_to_start_pattern) == type(None)):
            # wait until we're allowed to start
            while not os.path.exists(
                    self.ready_to_start_pattern.format(iteration)):
                time.sleep(1)
        # TODO test whether there's enough space in the destination
        # does this make sense for weird remote mounts?
        free_space = self.get_free_space()
        while not free_space > self.space_needed:
            time.sleep(1)
            free_space = self.get_free_space()
        return None
    def work(self, iter_list):
        for i in iter_list:
            self.start(i)
            self.iterate(i)
            self.finish(i)
        return None

