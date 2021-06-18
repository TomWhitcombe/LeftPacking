#include "SSE2LeftPack.h"
#include "SOA_AOS.h"

int main(int argc, char const *argv[])
{
    //remnants of the left pack test- yes I manually went though all 16 types. Sue me
    {
        float floats [4] = {1.0f,2.0f,3.0f, 4.0f};
        unsigned int mask[4] = { SELECTION_MASK(1,1,1,1) };

        __m128 __floats = _mm_load_ps(floats);
        __m128 __comps = _mm_load_ps((float*)mask);

        __m128 packed = _mm_leftPack_ps(__floats, __comps);
    }

    //Transpose test
    {
        __m128 xs = _mm_set_ps(7,5,3,1);
        __m128 ys = _mm_set_ps(8,6,4,2);

        vec2Out_t transposed = vec2_SOA2AOS(xs, ys);

        vec2Out_t transposedBack = vec2_AOS2SOA(transposed.a, transposed.b);
    }
    return 0;
}