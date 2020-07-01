#ifndef FAISS_ONEAPI_UTILS_KERNEL_HPP
#define FAISS_ONEAPI_UTILS_KERNEL_HPP

#include <exception>
#include <type_traits>

#include <CL/sycl.hpp>

namespace faiss::oneapi::utils
{
    struct kernel
    {
        virtual static bool is_device_supported(const cl::sycl::device& d) = 0;
        virtual static int optimization_level(const cl::sycl::device& d) = 0; 
        virtual static kernel create(cl::sycl::queue& q) = 0;
    };

    template<typename kernel_a_type, typename kernel_b_type>
    bool compare_kernels(const cl::sycl::device& d, const kernel_a_type& a, const kernel_b_type& b)
    {
        {
            constexpr const char* err_msg_not_derived = "Kernel types should be derived from `kernel`";
            static_assert(std::is_base_of(kernel, kernel_a_type), err_msg);
            static_assert(std::is_base_of(kernel, kernel_b_type), err_msg);
        }
        {
            constexpr const char* err_msg_not_dupported = "Kernel should support device";
            if(!a.is_device_supported(d)) std::system_error(err_msg_not_supported);
            if(!b.is_device_supported(d)) std::system_error(err_msg_not_supported);
        }
        return a.optimization_level(d) < b.optimization_level(d);
    }
}

#endif