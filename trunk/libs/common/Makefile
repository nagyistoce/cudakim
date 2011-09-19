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

INCLUDES = -I..

LIBNAME = libicgip_common.a

.PHONY: debug

../$(LIBNAME): error.o image2d.o
	$(LINKER) ../$(LIBNAME) $^ 

error.o: error.c error.h
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -c $< -o $@

image2d.o: image2d.c image2d.h
	$(COMPILER) $(COMPILER_FLAGS) $(INCLUDES) -c $< -o $@

debug: ../$(LIBNAME)

clean_obj:
	rm -f *.o

clean_bin:
	rm -f ../$(LIBNAME)

clean: clean_obj clean_bin
