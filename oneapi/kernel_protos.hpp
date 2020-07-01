#ifndef FAISS_ONEAPI_KERNEL_PROTOS_HPP
#define FAISS_ONEAPI_KERNEL_PROTOS_HPP    

#include <faiss/oneapi/utils/kernel.hpp>

namespace faiss::oneapi
{    
    template<typename type>
    struct l2_norms_kernel_proto : public kernel
    {
        virtual cl::sycl::event operator() (matrix<const type> data, type* const out) const = 0;
    };
}

#endif