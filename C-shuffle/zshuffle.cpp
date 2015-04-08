/************************************************************************
*
*  Copyright 2015 Johns Hopkins University
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
* Contact: turbulence@pha.jhu.edu
* Website: http://turbulence.pha.jhu.edu/
*
************************************************************************/



#include <stdlib.h>
#include <stdio.h>
#include <fftw3.h>
#include <algorithm>
#include <threadpool.h>

ptrdiff_t part1by2(ptrdiff_t x)
{
    ptrdiff_t n = x & 0x000003ff;
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n <<  8)) & 0x0300f00f;
    n = (n ^ (n <<  4)) & 0x030c30c3;
    n = (n ^ (n <<  2)) & 0x09249249;
    return n;
}

ptrdiff_t unpart1by2(ptrdiff_t z)
{
        ptrdiff_t n = z & 0x09249249;
        n = (n ^ (n >>  2)) & 0x030c30c3;
        n = (n ^ (n >>  4)) & 0x0300f00f;
        n = (n ^ (n >>  8)) & 0xff0000ff;
        n = (n ^ (n >> 16)) & 0x000003ff;
        return n;
}

class strided_to_cubbies_info
{
    public:
        ptrdiff_t cubbie_size;
        ptrdiff_t slice_size;
        float *a;
        float *b;
        int N;
        int n;
        int N0;
        int n0;
        int dim;
        int k;

        strided_to_cubbies_info(
                int N,
                int N0,
                int k,
                int dim,
                float *a,
                float *b)
        {
            this->cubbie_size = 8*8*8*dim;
            this->slice_size = ptrdiff_t(8)*ptrdiff_t(N)*ptrdiff_t(N)*dim;
            this->N = N;
            this->N0 = N0;
            this->n0 = N0 / 8;
            this->n = N / 8;
            this->k = k;
            this->dim = dim;
            this->a = a + k*this->slice_size;
            this->b = b;
        }
        ~strided_to_cubbies_info(){}
};

class array_2D_info
{
    public:
        ptrdiff_t N;
        float *a;

        array_2D_info(
                ptrdiff_t N,
                float *a)
        {
            this->N = N;
            this->a = a;
        }
        ~array_2D_info(){}
};

void array_to_cubbies(void *INFO);
void array_2D_to_zero(void *INFO);

extern "C"
{
    int shuffle_1CPU_plain(
            int N,
            int dim,
            float *a,
            float *b)
    {
        int n = N/8;
        ptrdiff_t z;
        ptrdiff_t cubbie_size = 8*8*8*dim;
        float *rz = fftwf_alloc_real(cubbie_size);
        float *at;
        for (ptrdiff_t k = 0; k < n; k++)
        for (ptrdiff_t j = 0; j < n; j++)
        for (ptrdiff_t i = 0; i < n; i++)
        {
            // first, copy data ptrdiff_to cubbie
            for (ptrdiff_t tk = 0; tk < 8; tk++)
            for (ptrdiff_t tj = 0; tj < 8; tj++)
            {
                at = a + (((k*8+tk)*N + (j*8+tj))*N + i*8)*dim;
                std::copy(at, at+8*dim, rz + (tk*8 + tj)*8*dim);
            }
            // now copy entire cubbie
            z = part1by2(k) | (part1by2(j) << 1) | (part1by2(i) << 2);
            std::copy(rz, rz + cubbie_size, b + z*cubbie_size);
        }
        fftwf_free(rz);
        return EXIT_SUCCESS;
    }

    int shuffle_1CPU(
            int N,
            int dim,
            float *a,
            float *b)
    {
        strided_to_cubbies_info stc(N, N, 0, dim, a, b);
        array_to_cubbies(&stc);
        return EXIT_SUCCESS;
    }

    int shuffle_threads(
            int N,
            int dim,
            float *a,
            float *b,
            int threads)
    {
        threadpool_t *pool = threadpool_create(threads, threads, 0);
        strided_to_cubbies_info *stc[threads];
        for (int t = 0; t < threads; t++)
        {
            stc[t] = new strided_to_cubbies_info(
                    N, N/threads, t*N/(8*threads), dim, a, b);
            threadpool_add(pool, &array_to_cubbies, stc[t], 0);
        }
        threadpool_destroy(pool, threadpool_graceful);
        for (int t = 0; t < threads; t++)
            delete stc[t];
        return EXIT_SUCCESS;
    }

    int array3D_to_zero_threads(
            int N0,
            int N1,
            int N2,
            int dim,
            float *a,
            int threads)
    {
        threadpool_t *pool = threadpool_create(threads, N0, 0);
        array_2D_info *a2i[N0];
        ptrdiff_t N = ptrdiff_t(N1)*ptrdiff_t(N2)*ptrdiff_t(dim);
        for (int t = 0; t < N0; t++)
        {
            a2i[t] = new array_2D_info(N, a + ptrdiff_t(t)*N);
            threadpool_add(pool, &array_2D_to_zero, a2i[t], 0);
        }
        threadpool_destroy(pool, threadpool_graceful);
        for (int t = 0; t < threads; t++)
            delete a2i[t];
        return EXIT_SUCCESS;
    }
}

void array_to_cubbies(void *INFO)
{
    strided_to_cubbies_info *info = (strided_to_cubbies_info*)(INFO);
    float *at;
    float *rz = fftwf_alloc_real(info->cubbie_size);
    ptrdiff_t z;
    for (ptrdiff_t k = 0; k < info->n0; k++)
    {
        for (ptrdiff_t j = 0; j < info->n; j++)
        for (ptrdiff_t i = 0; i < info->n; i++)
        {
            z = part1by2(k+info->k) | (part1by2(j) << 1) | (part1by2(i) << 2);
            // first, copy data ptrdiff_to cubbie
            for (ptrdiff_t tk = 0; tk < 8; tk++)
            for (ptrdiff_t tj = 0; tj < 8; tj++)
            {
                at = info->a + (((k*8+tk)*info->N + (j*8+tj))*info->N + i*8)*info->dim;
                std::copy(at, at+8*info->dim, rz + (tk*8 + tj)*8*info->dim);
            }
            // now copy entire cubbie
            std::copy(rz, rz + info->cubbie_size, info->b + z*info->cubbie_size);
        }
    }
    fftwf_free(rz);
}

void array_2D_to_zero(void *INFO)
{
    array_2D_info *info = (array_2D_info*)(INFO);
    std::fill_n(info->a, info->N, 0.0);
}

