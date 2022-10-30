# FiltersDemo

A simple application to try out different nonlinear edge preserving smoothing filters. It works with .bmp images currently, but i plan to extend it to different formats and 3D data structures.
The program uses CUDA for basic calculations and OpenGL for displaying the image. Currently the image is written into the GPU memory for calculation and smoothing, back to the CPU side and then back to the GPU again for rendering, which can be simplified by interoping between CUDA and OpenGL.
