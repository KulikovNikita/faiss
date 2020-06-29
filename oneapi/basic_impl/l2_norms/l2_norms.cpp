#include <faiss/oneapi/basic_impl/l2_norms.hpp>

static bool l2_norms_kernel::is_device_supported(const cl::sycl::device& d)
{
    return true;
}

int l2_norms_kernel::optimization_level(const cl::sycl::device& d) const
{
    //It's the baseline for other implementations
    return 0;
}