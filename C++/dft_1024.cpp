
// DFT N=1024 — oneAPI

#include <sycl/sycl.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <cmath>
#include <vector>
#include <iostream>
using namespace sycl;

#define N   1024
#define PI  3.14159265358979323846f

int main() {
    std::vector<float> x_real(N), x_imag(N);
    for (int n = 0; n < N; n++) {
        x_real[n] = std::cos(2.0f * PI * 8.0f * n / (float)N);
        x_imag[n] = 0.0f;
    }
    std::vector<float> X_real(N, 0.0f), X_imag(N, 0.0f);

#if defined(FPGA_EMULATOR)
    auto selector = ext::intel::fpga_emulator_selector_v;
#else
    auto selector = default_selector_v;
#endif

    queue q(selector);
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    {
        buffer<float> buf_xr(x_real.data(),  N);
        buffer<float> buf_xi(x_imag.data(),  N);
        buffer<float> buf_Xr(X_real.data(),  N);
        buffer<float> buf_Xi(X_imag.data(),  N);

        q.submit([&](handler &h) {
            auto xr = buf_xr.get_access<access::mode::read>(h);
            auto xi = buf_xi.get_access<access::mode::read>(h);
            auto Xr = buf_Xr.get_access<access::mode::write>(h);
            auto Xi = buf_Xi.get_access<access::mode::write>(h);
            h.parallel_for<class DFT_kernel>(range<1>(N), [=](id<1> idx) {
                int k = (int)idx[0];
                float sr = 0.0f, si = 0.0f;
                for (int n = 0; n < N; n++) {
                    float a = -2.0f * PI * (float)k * (float)n / (float)N;
                    sr += xr[n]*sycl::cos(a) - xi[n]*sycl::sin(a);
                    si += xr[n]*sycl::sin(a) + xi[n]*sycl::cos(a);
                }
                Xr[k] = sr; Xi[k] = si;
            });
        }).wait();
    }

    std::cout << "DFT bins > 1:\n";
    for (int k = 0; k < 20; k++) {
        float m = std::sqrt(X_real[k]*X_real[k] + X_imag[k]*X_imag[k]);
        if (m > 1.0f) std::cout << "  bin[" << k << "] = " << m << "\n";
    }
    std::cout << "DONE\n";
    return 0;
}
