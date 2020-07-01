#ifndef FAISS_ONEAPI_UTILS_REGISTRY_HPP
#define FAISS_ONEAPI_UTILS_REGISTRY_HPP

#include 

#include <CL/sycl.hpp>

#include <faiss/oneapi/utils/kernel.hpp>

namespace faiss::oneapi::utils
{
    template<typename abstract_kernel_type>
    struct registry
    {
        template<typename kernel_type>
        void add(const kernel_type)
    };
}

#endif