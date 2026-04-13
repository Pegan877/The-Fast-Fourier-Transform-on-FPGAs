
// FFT N=1024 — oneAPI

#include <sycl/sycl.hpp>
#include <ext/intel/fpga_extensions.hpp>
#include <cmath>
#include <vector>
#include <iostream>
using namespace sycl;

#define N      1024
#define LOG2_N 10
#define PI     3.14159265358979323846f

static unsigned int bit_rev(unsigned int x, int b) {
    unsigned int r = 0;
    for (int i = 0; i < b; i++) { r = (r<<1)|(x&1); x>>=1; }
    return r;
}

int main() {
    std::vector<float> a_real(N), a_imag(N);
    for (int n = 0; n < N; n++) {
        unsigned int rev = bit_rev((unsigned int)n, LOG2_N);
        a_real[rev] = std::cos(2.0f * PI * 8.0f * n / (float)N);
        a_imag[rev] = 0.0f;
    }

#if defined(FPGA_EMULATOR)
    auto selector = ext::intel::fpga_emulator_selector_v;
#else
    auto selector = default_selector_v;
#endif

    queue q(selector);
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << "\n";

    {
        buffer<float> buf_r(a_real.data(), N);
        buffer<float> buf_i(a_imag.data(), N);

        for (int s = 1; s <= LOG2_N; s++) {
            int m  = 1 << s;
            int m2 = m >> 1;
            float ang = -PI / (float)m2;

            q.submit([&](handler &h) {
                auto ar = buf_r.get_access<access::mode::read_write>(h);
                auto ai = buf_i.get_access<access::mode::read_write>(h);
                h.parallel_for<class FFT_stage>(range<1>(N/2), [=](id<1> idx) {
                    int p = (int)idx[0];
                    int g = p / m2, j = p % m2;
                    int u = g*m + j, v = u + m2;
                    float wr = sycl::cos(ang*(float)j);
                    float wi = sycl::sin(ang*(float)j);
                    float ur = ar[u], ui = ai[u];
                    float vr = ar[v]*wr - ai[v]*wi;
                    float vi = ar[v]*wi + ai[v]*wr;
                    ar[u]=ur+vr; ai[u]=ui+vi;
                    ar[v]=ur-vr; ai[v]=ui-vi;
                });
            }).wait();
        }
    }

    std::cout << "FFT bins > 1:\n";
    for (int k = 0; k < 20; k++) {
        float m = std::sqrt(a_real[k]*a_real[k] + a_imag[k]*a_imag[k]);
        if (m > 1.0f) std::cout << "  bin[" << k << "] = " << m << "\n";
    }
    std::cout << "DONE\n";
    return 0;
}
