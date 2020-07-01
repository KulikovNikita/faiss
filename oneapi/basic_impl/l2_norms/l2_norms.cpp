#include <faiss/oneapi/basic_impl/l2_norms.hpp>

namespace faiss::oneapi::basic_impl
{
    template<>
    l2_norms_kernel<float>::l2_norms_kernel(cl::sycl::queue& q_) : q(q_), 
        pref_vec_width(q_.get_device().get_info<cl::sycl::info::device::preferred_vector_width_float>())
        pref_wg_size(q.get_device().get_info<cl::sycl::info::device::max_compute_units>()) {};

    template<>
    l2_norms_kernel<float>::l2_norms_kernel(cl::sycl::queue& q_) : q(q_), 
        pref_vec_width(q_.get_device().get_info<cl::sycl::info::device::preferred_vector_width_double>())
        pref_wg_size(q.get_device().get_info<cl::sycl::info::device::max_compute_units>()) {};

    template<typename type>
    l2_norms_kernel<float> l2_norms_kernel<float>::create(cl::sycl::queue& q_)
    {
        return l2_norms_kernel<float>(q_);
    }

    template<typename type>
    static bool is_device_supported(const cl::sycl::device& d)
    {
        //basic implementation should support all hardware that can work with this type
        return (d.get_info<cl::sycl::info::device::preferred_vector_width_double>() != 0);
    }

    template<typename type>
    static int optimization_level(const cl::sycl::device& d)
    {
        //It is a baseline
        return 0;
    };

    template<typename type>
    cl::sycl::event operator() (matrix<const type> data, type* const out) const
    {
        //TODO: thin/thick dataset
        if(true)
        {
            return q.submit(
                [&](cl::sycl::handler& h)
                {
                    h.parallel_for<class l2_norms_basic_thin<type>>(
                        cl::sycl::range<1>(data.n),
                        [=](cl::sycl::id<1> idx)
                        {
                            const int id = idx[0];
                            const type* const row = &((data.data)[id * (data.stride)]);
                            type acc = 0;
                            for(int i = 0; i < (data.m); ++i)
                            {
                                acc[i] = (row[i] * row[i]);
                            }
                            out[id] = acc;
                        });
                });
        }
        return cl::sycl::event();
    }

    template struct l2_norms_kernel<double>;
    template struct l2_norms_kernel<float>;
}