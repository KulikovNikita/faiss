#ifndef FAISS_ONEAPI_UTILS_NATRIX
#define FAISS_ONEAPI_UTILS_NATRIX

#include <CL/sycl.hpp>

namespace faiss::oneapi::utils
{
    template<typename type>
    struct matrix
    {
        //constructors
        matrix(const matrix<type>& mat);
        matrix(type* ptr, size_t m, size_t n);
        matrix(type* ptr, size_t m, size_t n, size_t stride);
        //data section
        const size_t m, n, stride;
        fp_type* const data;
    };

    //using anonymous namespace to hide functions to except collisions
    namespace
    {
        template<typename type>
        matrix::matrix(const matrix<type>& mat) : m(mat.m), n(mat.n), stride(mat.stride), data(mat.data) {};
        template<typename type>
        matrix::matrix(type* data_, size_t m_, size_t n_, size_t stride_) : m(m_), n(n_), stride(stride_), data(data_) {};
        template<typename type>
        matrix::matrix(type* data_, size_t m_, size_t n_) : m(m_), n(n_), stride(m_), data(data_) {};
    };
}

#endif