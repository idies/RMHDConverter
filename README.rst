RMHD converter
--------------

Code to convert RMHD data from p3DFFT specific Fourier space format to
z-indexed chunks of 8x8x8x2 cubbies.
This code is pretty application specific, although some parts may be
reusable for other tasks.

Note: I am now aware that there are several options for queueing tasks
in python, but my particular problem was so simple that I just did
something myself and it worked.
