CFLAGS = -Wall \
		 -O3
LDFLAGS = -L/home/clalescu/ext_installs/lib \
		  -L./threadpool
INCLUDE = -I/home/clalescu/ext_installs/include \
		  -I./threadpool/src

LDLIBS = -lfftw3 \
		 -lfftw3f

all:
	g++ \
		-shared \
		-fPIC \
		${CFLAGS} \
		${INCLUDE} \
		${LDFLAGS} \
		zshuffle.cpp \
		-o libzshuffle.so \
		${LDLIBS} \
		-Wl,--whole-archive ./threadpool/libthreadpool.a -Wl,--no-whole-archive

