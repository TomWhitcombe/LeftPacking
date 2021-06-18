#pragma once
#include <xmmintrin.h>

typedef struct vec2Out
{
    __m128 a,b;
} vec2Out_t;

static vec2Out_t vec2Transpose(__m128 a, __m128 b)
{
    vec2Out_t out;
    __m128 aabb = _mm_shuffle_ps(a,b, _MM_SHUFFLE(1,0,1,0));
    __m128 aabb2 = _mm_shuffle_ps(a,b, _MM_SHUFFLE(3,2,3,2));
    out.a = _mm_shuffle_ps(aabb, aabb, _MM_SHUFFLE(3,1,2,0));
    out.b = _mm_shuffle_ps(aabb2, aabb2, _MM_SHUFFLE(3,1,2,0));

    return out;
}