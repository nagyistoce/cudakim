--------------
 Introduction
--------------
These example programs have been written for the graduate course 
Data-parallel computing (dpc) held at Aarhus University. The source
code has been written by Christian P. V. Christoffersen (cpvc@cs.au.dk).

The example programs included have been tested on:
- OS X 10.6 and 10.7, with XCode installed
- Ubuntu 10.10, 64 bit

---------------------------
 Installation of dev tools
---------------------------
To be able to compile and execute CUDA programs you must first install
Nvidia's CUDA 4.0 developer tools: dev driver, toolkit, and optionally
the SDK.

Two of the examples use the cutil library which is part of the SDK
so you must install the SDK to be able to compile these examples.

If you are using one of the department's machines with a GTX-480
graphics card installed then the dev driver and toolkit should already
be installed.

If you want to use your own machine, then Nvidia's CUDA 4.0 developer
tools can be found on the following web page:
    http://developer.nvidia.com/cuda-toolkit-40
We are currently using version 4.0.17.

The department uses Ubuntu 10.10 and 11.04 with a ppa repository. If
you want the same setup then follow the instructions on the web site:
    https://launchpad.net/~aaron-haviland/+archive/cuda-4.0

Note that the ppa description above installs CUDA into /usr instead
of Nvidia's default location /usr/local/cuda.

The department's machines do not include an installation of the SDK.
You must install this yourself. The default install path of the SDK is
the following sub folder of your home dir: ~/NVIDIA_GPU_Computing_SDK.

To install the SDK on the department's machines do the following:
    wget http://developer.download.nvidia.com/compute/cuda/4_0/sdk/gpucomputingsdk_4.0.17_linux.run
    sh gpucomputingsdk_4.0.17_linux.run
    cd ~/NVIDIA_GPU_Computing_SDK/C
    make
    cd

To test if you have installed and compiled the SDK correctly execute:
    ~/NVIDIA_GPU_Computing_SDK/C/bin/linux/release/deviceQuery

The deviceQuery program should report:
    CUDA Driver Version = 4.0,
    CUDA Runtime Version = 4.0,
    NumDevs = 1, Device = GeForce GTX 480

If the versions are wrong please report this. If the device is
wrong you are not using one of the machines with a GTX-480
graphics card.

--------------------
 Build instructions
--------------------
The examples all use some common libraries, which must be compiled before
each of the individual examples are compiled. To compile the libraries
execute the following command: 'cd libs; make; cd ..'

When the libraries have been compiled each of the example programs
located in the other folders can be compiled. To compile for
example the first program, execute the command: 'cd exercise01; make; cd ..'

------------
 Exercise01
------------
This program illustrates how to write a simple CUDA program. It loads
an image file from disk, thresholds each pixel value, and saves the
result to disk. To run the program on the included image execute 'make test'
in the example directory. You can see how to use ImageMagick to convert
images to raw inside the Makefile.

The goal of this exercise is to ensure that you have installed the dev
driver and toolkit correctly and to show how to program, compile, link,
and execute a simple CUDA program.

------------
 Exercise02
------------
This exercise covers a generally applicable matrix multiplication
implementation, using the full hierarchical threading model and
memory tiling optimizations. Furthermore the exercise introduces the
concept of memory caching through the texture abstraction. It also
makes sure that you have installed the SDK correctly.

Edit the source file matrixmul_device.cu to complete the functionality
of the matrix multiplication on the device. Inside the code files the
"/// *** INSERT CODE ***" string should be filled with your code. The
two matrices could be any size, but the resulting matrix is guaranteed
to have a number of elements less than 64,000.

Please read through all the sub-exercises before you start coding, as
some of them reuse code. This could help you organize the code so
it can be reused.

Exercise 2.1:
Implement the Simple Matrix Multiplication (figure 4.6). Measure the
performance. 
NOTE: As of the GTX 480 (as you have in Stibitz) the global memory
access is cached, so this simple strategy is likely to perform rather
well. If you run this code on older hardware it is very slow...


Exercise 2.2: Tiled Matrix Multiplication.
Implement a kernel that uses shared memory and tiling to accelerate
the calculations (figure 5.7). Does it provide an acceleration over
the simpler approach from exercise 2.1?

Exercise 2.3 (OPTIONAL): Matrix multiplication using cached (textured) memory.
Convert the non tiled version of the matrix multiplication and use
texture cached memory to accelerate the calculations. Use the
following code to bind and use texture memory:

// global scope
texture<float, 1> tex;

// in function
const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
cudaBindTexture(0, &tex, dataDevPtr, &desc, num_bytes);
... kernel launch ...
cudaUnbindTexture(tex);

// in kernel
float val = tex1Dfetch(tex, index);

For further information go to: http://www.drdobbs.com/cpp/218100902,
press print and search for the example code using cudaBindTexture.

Do we need textures for caching on the GTX480?

Exercise 2.4:
Compare the execution speed of the three previous programming
exercises by trying different sizes of matrices and making time
measurements of the execution speed. Print these in a graph e.g. by
using gnuplot. For help on timing the kernels see chapter 2 of
Nvidia's best practice guide.

Exercise 2.5:
Use the nvcc compiler to inspect the amount of registers and shared
memory used by the different kernels. Present your findings in a table
and discuss them. Add --ptxas-options="-v" to the end of the $(NVCC)
line in the Makefile and recompile using make to see the resource
usage of your kernels.

Exercise 2.6:
Use the CUDA Compute Visual profiler (computeprof) installed by the
CUDA Toolkit to compare the different kernels. Explain and discuss the
profiler output.
