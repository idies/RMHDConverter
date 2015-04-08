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
import os

from file_mover import FileWorkerBase

def generate_data_3D(
        n,
        dtype = np.complex128,
        p = 1.5):
    """
    generate something that has the proper shape
    """
    assert(n % 2 == 0)
    a = np.zeros((n, n, n/2+1), dtype = dtype)
    a[:] = np.random.randn(*a.shape) + 1j*np.random.randn(*a.shape)
    k, j, i = np.mgrid[-n/2+1:n/2+1, -n/2+1:n/2+1, 0:n/2+1]
    k = (k**2 + j**2 + i**2)**.5
    k = np.roll(k, n//2+1, axis = 0)
    k = np.roll(k, n//2+1, axis = 1)
    a /= k**p
    a[0, :, :] = 0
    a[:, 0, :] = 0
    a[:, :, 0] = 0
    ii = np.where(k == 0)
    a[ii] = 0
    ii = np.where(k > n/3)
    a[ii] = 0
    return a

def padd_with_zeros(
        a,
        n,
        odtype = None):
    if (type(odtype) == type(None)):
        odtype = a.dtype
    assert(a.shape[0] <= n)
    b = np.zeros((n, n, n/2 + 1), dtype = odtype)
    m = a.shape[0]
    b[     :m/2,      :m/2, :m/2+1] = a[     :m/2,      :m/2, :m/2+1]
    b[     :m/2, n-m/2:   , :m/2+1] = a[     :m/2, m-m/2:   , :m/2+1]
    b[n-m/2:   ,      :m/2, :m/2+1] = a[m-m/2:   ,      :m/2, :m/2+1]
    b[n-m/2:   , n-m/2:   , :m/2+1] = a[m-m/2:   , m-m/2:   , :m/2+1]
    return b

class FileGenerator(FileWorkerBase):
    def __init__(
            self,
            dst_dir = None,
            ready_to_start_pattern = None,
            job_done_pattern = None,
            space_needed = 0,
            dst_pattern = None,
            n = None,
            N = None,
            p = 2):
        super(FileGenerator, self).__init__(
            ready_to_start_pattern = ready_to_start_pattern,
            dst_dir = dst_dir,
            job_done_pattern = job_done_pattern,
            space_needed = space_needed)
        self.n = n
        self.N = N
        self.p = p
        self.dst_pattern = dst_pattern
        return None
    def iterate(self, iteration):
        if type(self.dst_pattern) == type('string'):
            Kdata = generate_data_3D(
                self.n,
                p = self.p,
                dtype = '>c8')
            Kdata.T.copy().tofile(self.dst_pattern.format(iteration))
        elif type(self.dst_pattern) == type([]):
            for dp in self.dst_pattern:
                Kdata = generate_data_3D(
                    self.n,
                    p = self.p,
                    dtype = '>c8')
                Kdata.T.copy().tofile(
                    os.path.join(self.dst_dir, dp.format(iteration)))
        return None

