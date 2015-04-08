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
import matplotlib.pyplot as plt
import subprocess

def slice_plotter(n0 = 4, n1 = 8):
    ax = plt.figure(figsize = (8, 8)).add_axes([.0, .0, 1., 1.])
    ax.set_axis_off()
    for i in range(n0, n1, 4):
        print('plotting frame ', i)
        data = np.load(
            '/export/scratch0/clalescu/RMHD/2D_slices/data_rs2_u_t{0:0>4x}.npy'.format(i))
        for j in range(3):
            for k in range(2):
                ax.cla()
                ax.set_axis_off()
                ax.imshow(data[j, :, :, k])
                plt.gcf().savefig(
                    'figs/u{0}_{1}_t{2:0>4}.png'.format(j, k, i),
                    dpi = max(data[j, :, :, k].shape)//8,
                    format = 'png')
    return None


generate_png = True
generate_gif = False

if generate_png:
    slice_plotter(n0 = 128, n1 = 0x280)

if generate_gif:
    for j in range(3):
        for k in range(2):
            subprocess.call(['convert',
                             'figs/u{0}_{1}_t*.png'.format(j, k),
                             'figs/u{0}_{1}.gif'.format(j, k)])



