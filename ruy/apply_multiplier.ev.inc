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


std::int32_t Multiply_better(
    std::int32_t x, std::int32_t mul, int shift)
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

        typedef signed long long SLL;
        if (show_data_bool) show_data("before scaling {");

        SLL acc = SLL(x);    // Assumed to be an integer already.
        acc *= shift;
        x = double(LL_ROUND(acc,shift));

        if (show_data_bool) show_data("after scaling }");

        return x;
        } break;
    }
}

#define MultiplyByQuantizedMultiplier(A,B,C) \
    Multiply_better(A,B,C)
