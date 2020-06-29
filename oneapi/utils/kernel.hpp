#ifndef FAISS_ONEAPI_UTILS_KERNEL_HPP
#define FAISS_ONEAPI_UTILS_KERNEL_HPP

#include <type_traits>

#include <CL/sycl.hpp>

namespace faiss::oneapi::utils
{
    struct kernel
    {
        static bool is_device_supported(const cl::sycl::device& d) const;
        int optimization_level(const cl::sycl::device& d) const; 
    };

    template<typename kernel_a_type, typename kernel_b_type>
    bool operator< (const kernel_a_type& a, const kernel_b_type& b)
    {
        constexpr const char* err_msg = "Kernel types should be derived from `kernel`";
        static_assert(std::is_base_of(kernel, kernel_a_type), err_msg);
        static_assert(std::is_base_of(kernel, kernel_b_type), err_msg);
        return a.optimization_level() < b.optimization_level();
    }
}

#endif