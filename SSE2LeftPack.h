#pragma once
#include <xmmintrin.h>
/*
    Left Packing based on the SSE2 implementation from Andreas Fredrikson's GDC talk:
    https://www.gdcvault.com/play/1022248/SIMD-at-Insomniac-Games-How

    In it Andreas' talks about using move distances to figure out the selection masks. I didn't really follow that in all 
    honesty. Instead, I worked out the selection masks via Eyeball 1.0. It was actually quite a cathartic puzzle to solve for the
    16 different combinations. I've left this here to show my work in each of them.

    Whenever I work with SIMD I conceptualize it more easily in index order- ABCD. But modern PCs being little endian don't do that,
    and most masks are expected DCBA. It's probably a habbit I should get out of, but that's how my brain thinks at the moment.
    Because of this the values in the sections below aren't displayed in the correct index order-the index is the mirrored value
    eg- 0010 isn't index 2, but index 4 (0100).
*/

// ================
// 0000
// ABCD

// BCDA
// 0000
// ABCD

// CDBA
// 0000

// ================
// 0001 Conceptual Index (actual is 1000)
// ABCD value

// BCDA val rot 1
// 0010 first select mask
// ABDD select into value, alias tempVal

// DDAB rotate tempVal 2
// 1000 select into tempVal
// DBDD BOOM
// You may ask what about the BDD? Well that gets discarded by popcnt of the mask- wonderful

// ================
// 0010
// ABCD value

// BCDA rot 1
// 0000 select mask 0
// ABCD None are selected, but this is still tempVal

// CDAB rotate tempVal twice
// 1000 select mask 1 into tempVal
// CBCD BOOM

// ================
// 0011
// ABCD

// BCDA
// 0000 select 0
// ABCD apply select mask to val, alias tempVal

// CDAB rotate tempVal twice
// 1100 select mask
// CDCD BOOM

// ================
// 0100
// ABCD

// BCDA
// 1000 select 0
// BBCD apply select mask, alias result to tempVal- you get the picture

// CDBB rot 2
// 0000 select 1

// BBCD

// ================
// 0101
// ABCD

// BCDA
// 1000
// BBCD

// CDBB
// 0100

// BDCD

// ================
// 0110
// ABCD

// BCDA
// 1100
// BCCD

// CDBC
// 0000

// BCCD

// ================
// 0111
// ABCD

// BCDA
// 1110
// BCDD

// DDBC
// 0000

// BCDD

// ================
// 1000
// ABCD

// BCDA
// 0000
// ABCD

// CDAB
// 0000

// ABCD
// ================
// 1001
// ABCD

// BCDA
// 0000
// ABCD

// CDAB
// 0100

// ADCD

// ================
// 1010
// ABCD

// BCDA
// 0100
// ACCD

// CDAC
// 0000

// ================
// 1011
// ABCD

// BCDA
// 0110
// ACDD

// DDAC
// 0000

// ACDD

// ================
// 1100
// ABCD

// BCDA
// 0000
// ABCD

// CDAB
// 0000

// ABCD
// ================
// 1101
// ABCD

// BCDA
// 0010
// ABDD

// DDAB
// 0000

// ================
// 1110
// ABCD

// BCDA
// 0000
// ABCD

// CDAB
// 0000

// ABCD

// ================
// 1111
// BCDA
// 0000
// ABCD

// CDAB
// 0000

// ABCD

// ================

#define SELECTION_MASK(a, b, c, d) 0xFFFFFFFF * a, 0xFFFFFFFF * b, 0xFFFFFFFF * c, 0xFFFFFFFF * d

static __m128 _mm_leftPack_ps(__m128 value, __m128 mask)
{
    static unsigned int selectionMasksLUTData[16 * 8] =
        {
                                                                    //Inx  Conceptual Index
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0000 0000
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0001 1000
            SELECTION_MASK(1, 0, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0010 0100
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0011 1100
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(1, 0, 0, 0), //0100 0010
            SELECTION_MASK(0, 1, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0101 1010
            SELECTION_MASK(1, 1, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0110 0110
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(0, 0, 0, 0), //0111 1110
            SELECTION_MASK(0, 0, 1, 0), SELECTION_MASK(1, 0, 0, 0), //1000 0001
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(0, 1, 0, 0), //1001 1001
            SELECTION_MASK(1, 0, 0, 0), SELECTION_MASK(0, 1, 0, 0), //1010 0101
            SELECTION_MASK(0, 0, 1, 0), SELECTION_MASK(0, 0, 0, 0), //1011 1101
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(1, 1, 0, 0), //1100 0011
            SELECTION_MASK(0, 1, 1, 0), SELECTION_MASK(0, 0, 0, 0), //1101 1011
            SELECTION_MASK(1, 1, 1, 0), SELECTION_MASK(0, 0, 0, 0), //1110 0111
            SELECTION_MASK(0, 0, 0, 0), SELECTION_MASK(0, 0, 0, 0)  //1111 1111
        };

    int lutINDX = _mm_movemask_ps(mask);

    __m128 selector0 = _mm_load_ps((float*)&selectionMasksLUTData[lutINDX * 8]);
    __m128 selector1 = _mm_load_ps((float*)&selectionMasksLUTData[(lutINDX * 8) + 4]);

    __m128 rotatedOnce = _mm_shuffle_ps(value, value, _MM_SHUFFLE(0,3,2,1));
    __m128 result = _mm_or_ps(_mm_and_ps(rotatedOnce, selector0), _mm_andnot_ps(selector0, value));

    __m128 rotatedTwice = _mm_shuffle_ps(result, result, _MM_SHUFFLE(1,0,3,2));
    result = _mm_or_ps(_mm_and_ps(rotatedTwice, selector1), _mm_andnot_ps(selector1, result));

    return result;
}

//#undef MASK_ONE
//#undef SELECTION_MASK