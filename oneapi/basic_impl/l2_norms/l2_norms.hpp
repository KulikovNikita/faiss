#ifndef FAISS_ONEAPI_BASIC_IMPL_L2_NORMS_HPP
#define FAISS_ONEAPI_BASIC_IMPL_L2_NORMS_HPP

#include <CL/sycl.hpp>

#include <faiss/oneapi/kernel_protos.hpp>

namespace faiss::oneapi::basic_impl
{
    template<typename type>
    struct l2_norms_kernel : public l2_norms_kernel_proto<type>
    {
        private:
            l2_norms_kernel(cl::sycl::queue& q);
        private:
            const size_t pref_vec_width, pref_wg_size;
            cl::sycl::queue& q;
    };
}

#endif