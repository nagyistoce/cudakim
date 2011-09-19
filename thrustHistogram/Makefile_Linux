NVCC=nvcc
CC=gcc
LINK=g++

INCLUDES = -I. -I../libs -I/usr/local/cuda/include
LIBS = -L/usr/local/cuda/lib -lcuda -lcudart
GL_LIBS = -lGLEW

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
    INCLUDES += -I/opt/local/include
    LIBS += -L/opt/local/lib
    GL_LIBS += -framework glut -framework OpenGL 
    CUDA_SDK=/Developer/GPU\ Computing/C
else
    INCLUDES += -I/usr/include -I/usr/local/include
    LIBS += -L/usr/lib -L/usr/local/lib
    GL_LIBS += -L/usr/X11/lib -lglut -lGL -lGLU
    CUDA_SDK=$(HOME)/NVIDIA_GPU_Computing_SDK/C
endif

# detect OS
ARCHUPPER = $(shell uname -m 2>/dev/null | tr [:lower:] [:upper:])
# 'linux' is output for Linux system, 'darwin' for OS X
ARCH := $(strip $(findstring X86_64, $(ARCHUPPER)))
ifneq ($(ARCH),)
    LIB_ARCH = x86_64
    NVCCFLAGS += -m64
    ifneq ($(DARWIN),)
        CXX_ARCH_FLAGS += -arch x86_64
    else
        CXX_ARCH_FLAGS += -m64
    endif
else
    LIB_ARCH = i386
    NVCCFLAGS += -m32
    ifneq ($(DARWIN),)
        CXX_ARCH_FLAGS += -arch i386
    else
        CXX_ARCH_FLAGS += -m32
    endif
endif


INCLUDES += -I$(CUDA_SDK)/common/inc
LIBS += -L$(CUDA_SDK)/lib -lcutil_$(LIB_ARCH)

BINARY=histogram

$(BINARY): $(BINARY).o
	$(LINK) $(CXX_ARCH_FLAGS) $^ -o $(BINARY) $(LIBS) $(GL_LIBS)

#BmpUtil.o: BmpUtil.cpp
#	$(CC) $(CXX_ARCH_FLAGS) -c $(INCLUDES) $< -o $@

histogram.o: histogram.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCLUDES) $< -o $@

clean:
	rm -f *.o
	rm -f $(BINARY)
