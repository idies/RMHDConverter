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
import sys
import os
sys.path.append('/home/clalescu/repos/personal/py/lib')
import chichi_misc as cm

N = 2048

iteration = 144000
rdata = {}
field_name = ['ux', 'uy', 'uz', 'bx', 'by', 'bz']

for k in [2, 3, 5, 6]:
    fname = os.path.join(
        '/datascope/tdbrmhd04/real_space/b2048n/',
        'R{0:0>6}QNP.{1}'.format(iteration, k))
    rdata[field_name[k-1]] = np.memmap(
            fname,
            dtype = '<f4',
            mode = 'r',
            shape = (N, N, N))

data = {'uy': [],
        'uz': [],
        'by': [],
        'bz': [],
        'ix': [],
        'iy': [],
        'iz': []}

def pseudo_random_indices(
        N = 1024, n = 8, rseed = None):
    if rseed:
        np.random.seed(rseed)
    return (np.arange(0, N, N//n).astype(int) +
            np.random.randint(N/n, size=n)) % N

def append_data(i0, i1, i2):
    for j1 in range(8):
        for j0 in range(8):
            data['ix'].append(i2[:, j1, j0])
            data['iy'].append(i1[:, j1, j0])
            data['iz'].append(i0[:, j1, j0])
            for key in ['uy', 'uz', 'by', 'bz']:
                data[key].append(rdata[key][i0[:, j1, j0],
                                            i1[:, j1, j0],
                                            i2[:, j1, j0]].copy())

i0, i1, i2 = np.meshgrid(
    range(2048),
    pseudo_random_indices(N = 2048),
    pseudo_random_indices(N = 2048),
    indexing = 'ij')
append_data(i0, i1, i2)

i0, i1, i2 = np.meshgrid(
    range(2048),
    pseudo_random_indices(N = 2048),
    pseudo_random_indices(N = 2048),
    indexing = 'ij')
append_data(i1, i2, i0)

i0, i1, i2 = np.meshgrid(
    range(2048),
    pseudo_random_indices(N = 2048),
    pseudo_random_indices(N = 2048),
    indexing = 'ij')
append_data(i2, i0, i1)

for k in data.keys():
    np.save('RMHD_1D_cuts_' + k, np.array(data[k]))

