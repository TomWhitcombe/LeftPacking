#include "SSE2LeftPack.h"

int main(int argc, char const *argv[])
{
    float floats [4] = {1.0f,2.0f,3.0f, 4.0f};
    unsigned int mask[4] = { SELECTION_MASK(1,1,1,1) };

    __m128 __floats = _mm_load_ps(floats);
    __m128 __comps = _mm_load_ps((float*)mask);

    __m128 packed = _mm_leftPack_ps(__floats, __comps);
    return 0;
}