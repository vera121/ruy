/*
The code below is to change rounding behavior for a convolution.

In EV hardware an accumulated value in a convolution or innerproduct is reduced
to fit into the destination blob size (e.g. 8 bits).

To implement the reduction, the accumulator is effectively multiplied by a
floating-point number with a 15-bit mantissa.  Integer multiplication and shift
implement this multiplication, as the hardware does not have floating point:

- A s32 x s16 multiplication is done, producing an s48-bit result
- The s48 result is shifted right and rounded symmetrically (also called round-to-even).

Next:
- If there is a zero point, it is added,
  and the result saturated in the range [0,2*N-1] for an N-bit blob
- otherwise, relu bounds, if any, are applied before storing into the destination blob.

The computation of the 15-bit mantissa is derived from a floating-point number F.
F is the scale of the output blob divided by the scale of the accumulator.
Note: the EV notion of scale is the inverse of Tensorflow; i.e. the floating
point number represented by a pixel is F = (pixel - zero_point) / scale .
Thus in Synopsys EV,

    F = input_scale * (double)weight_scale / output_scale;
        ^^^ compute accumulator scale ^^^

The function normalize_fractional below takes the floating point number and computes
the 15-bit mantiss and its accompanying shift, taking care to produce a number
<= 32767 and a shift >= 1, in case hardware doesn't support a shift of 0.
In addition if the computed integer is even it's shifted right to remove 0s
solely for representational efficiency.

So, in summary:
- compute a floating-point number that reduces the accumulator to the desired
  output scale
- Convert the floating-point number to a 15-bit integer and 6-bit shift
- Multiply the accumulator by the integer and shift with convergent rounding

The macro
    define LL_ROUND(X,shift) / (unbiased) round-to-even / \
       ((X + ((X >> (shift)) & 1) + (LLSHL1(shift-1)-1)) >> (shift))
implements round-to-even -- i.e., unbiased (convergent) rounding.

The environment variable
    set TF_QUANTIZED_ROUND=EV
engages this alternative rounding.

 */



typedef double Scale_type;
#include <stdlib.h>
#include <cmath>
#include <stdint.h>
#include <string.h>

static void normalize_fractional(Scale_type F, unsigned &mpy, unsigned &shift) {
    // Adapted from python code in evgencnn.
    int frac_bits = 16;
    Scale_type hi = 0.5;
    int nudge_power = frac_bits;
    // Due to symmetric rounding, >= 32767.5 rounds to 32768, which is invalid
    // as a 16-bit signed number.
    // So the high value should be shifted by 32767.49/32768 so rounding
    // will produce at most 32767.  Nudge avoids that.
    unsigned two_to_nudge = 1<<nudge_power;
    hi *= (Scale_type(two_to_nudge)-0.51)/two_to_nudge;
    Scale_type lo = hi/2;
    int frac_adjust_shift = 0;
    F = fabs(F);
    Scale_type oldF = F;
    while (F >= hi) {
    frac_adjust_shift -= 1; F /= 2;
    }
    while (F < lo) {
    frac_adjust_shift += 1; F *= 2;
    }

    int max_shift = 63;
    while (frac_bits + frac_adjust_shift > max_shift) {
    frac_adjust_shift--;
    }
    int total_shift = frac_bits + frac_adjust_shift;
    0 && printf("F=%f fas=%d\n",F,frac_adjust_shift);
    0 && printf("newF=%f\n",oldF*(1<<total_shift));
    mpy = std::round(oldF * (1<<frac_bits) * (1<<frac_adjust_shift));
    // Now if mpy is even, divide by 2 and reduce the shift.
    shift = frac_bits + frac_adjust_shift;
    const int MINSHIFT = 1;  // Not knowing whether HW likes shift of 0, we make min 1.
    while ((mpy & 1) == 0 && shift > MINSHIFT) {
        // The end result is an odd fractional.
        mpy >>= 1; shift -= 1;
    }
    }


std::int32_t Multiply_better(
    std::int32_t x, std::int32_t mul, int shift,
    const Scale_type scale)
    {
    enum Rmode { R_double_round, R_ev_round };
    auto tell = []() {
    const char* QR = getenv("TF_QUANTIZED_ROUND");
    if (QR == 0) return R_double_round;
    return
        strcmp(QR,"EV")==0?R_ev_round:
        (printf("Unrecognized rounding mode %s\n",QR), R_double_round);
    };
    static const Rmode QR = tell();
    static bool show_data_bool = getenv("TF_SHOW_DATA") != 0;
    auto show_data = [&](const char *when) {
    printf("Data %s\n",when);
    printf("x = %f\n",x);
    return 0;
    };
    switch(QR) {
        case R_double_round: {
        MultiplyByQuantizedMultiplier(x, mul, shift);
        } break;
    case R_ev_round: {
        #define LLSHL1(x) (1LL<<(x))
        #define LL_ROUND(X,shift) /* (unbiased) round-to-even */ \
        ((X + ((X >> (shift)) & 1) + (LLSHL1(shift-1)-1)) >> (shift))
        unsigned mpy,shift;
        // Produces 15-bit mantissa and an exponent.  The mantissa is
        // thus less precise than that of a 32-bit floating-point number.
        normalize_fractional(scale,mpy,shift);
        printf("    Layer: round mode %d by %18.15f = mpy %d shift %d\n",
        QR,scale,mpy,shift);
        typedef signed long long SLL;
        if (show_data_bool) show_data("before scaling {");

        SLL acc = SLL(x);    // Assumed to be an integer already.
        acc *= mpy;
        x = double(LL_ROUND(acc,shift));

        if (show_data_bool) show_data("after scaling }");

        return x;
        } break;
    }
    }

#define MultiplyByQuantizedMultiplier(A,B,C) \
    Multiply_better(A,B,C, out_scal)
