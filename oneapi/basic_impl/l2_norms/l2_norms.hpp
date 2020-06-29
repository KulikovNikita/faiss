#ifndef FAISS_ONEAPI_BASIC_IMPL_L2_NORMS_HPP
#define FAISS_ONEAPI_BASIC_IMPL_L2_NORMS_HPP

#include <CL/sycl.hpp>

#include <faiss/oneapi/utils/kernel.hpp>

namespace faiss::oneapi::basic_impl
{
    using namespace faiss::oneapi::utils;

    class l2_norms_kernel : faiss::oneapi::utils::kernel
    {
        //Overriding of kernel functionality

    };
}

#endif