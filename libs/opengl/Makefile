COMPILER = gcc
#LINKER = ar -r -cs 
LINKER = ld -r -o 

COMPILER_FLAGS = -ansi -Wall
ifeq ($(COMPILER),gcc)
COMPILER_FLAGS += -std=c99
endif

ifeq ($(MAKECMDGOALS),debug)
COMPILER_FLAGS += -g -ggdb
endif

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
# 'linux' is output for Linux system, 'darwin' for OS X
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
# MAC OSX	
INCLUDES = -I..
LIBS     = -framework OpenGL -framework GLUT
else
INCLUDES = -I/usr/local/include -I..
LIBS     = -L/usr/local/lib -lGL -lGLU -lglut
endif

LIBNAME = libicgip_opengl.a

.PHONY: debug

../$(LIBNAME): opengl_utilities.o glsl_shader.o image2d_opengl.o
	$(LINKER) ../$(LIBNAME) $^ 

opengl_utilities.o: opengl_utilities.c opengl_utilities.h
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -c $< -o $@

glsl_shader.o: glsl_shader.c glsl_shader.h
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -c $< -o $@

image2d_opengl.o: image2d_opengl.c image2d_opengl.h
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -c $< -o $@

debug: ../$(LIBNAME)

clean_obj:
	rm -f *.o

clean_bin:
	rm -f ../$(LIBNAME)

clean: clean_obj clean_bin
