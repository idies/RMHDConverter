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

