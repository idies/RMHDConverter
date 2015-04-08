
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



