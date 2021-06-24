/* Copyright 2020 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ruy/apply_multiplier.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace ruy {
namespace detail {

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Warning: this code is not meant to be bit-exact-normative.
// Please refer to the class comment of ruy::MulParams, in mul_params.h.
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Simplified multiplier application function
//
// Double rounding and symmetric rounding are removed compared to reference.
// Double rounding seems unnecessary and can complicate implementations.
// Symmetric rounding also adds implementation complexity.
//
// Composed of a single rounding shift right and can lead to more HW
// friendly implementations.
//
// On NEON this can be translated to a SQDMULH + rounding shift right sequence.
// The use of SQDMULH rather than SQRDMULH gives a result that is
// equivalent to a single rounded shift since the truncating shift of SQDMULH
// can be combined with the rounding right shift via the formula (for k>=1):
//  ((x>>31)+(1<<(k-1)))>>k = (x + (1<<(30+k))>>(31+k)
//
// Preconditions:
// - quantized_multiplier >= 0
// - shift is -31 to +7 (negative for right shift)

// EV CUSTOMIZATION: add "TF" prefix
std::int32_t TFMultiplyByQuantizedMultiplier(std::int32_t x,
                                           std::int32_t quantized_multiplier,
                                           int shift) {
  RUY_CHECK_GE(shift, -31);
  RUY_CHECK_LE(shift, 7);

  int total_shift = 31 - shift;

  std::int64_t x_64(x);
  std::int64_t quantized_multiplier_64(quantized_multiplier);
  std::int64_t round = (int64_t)1 << (total_shift - 1);
  int64_t result = x_64 * quantized_multiplier_64 + round;
  result = result >> total_shift;

  RUY_DCHECK_GE(result, std::numeric_limits<std::int32_t>::lowest());
  RUY_DCHECK_LE(result, std::numeric_limits<std::int32_t>::max());

  return static_cast<std::int32_t>(result);
}


// <----SNPS EV rounding
typedef double Scale_type;
#include <string.h>

std::int32_t MultiplyByQuantizedMultiplier(
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
    printf("x = %d\n",x);
    return 0;
    };

    if (show_data_bool) printf("mul=%d, shift=%d\n");

    switch(QR) {
        case R_double_round: {
        	if (show_data_bool) show_data("before scaling {");
        	x = TFMultiplyByQuantizedMultiplier(x, mul, shift);
        	if (show_data_bool) show_data("after scaling }");
        	} break;
		case R_ev_round: {
			#define LLSHL1(x) (1LL<<(x))
			#define LL_ROUND(X,shift) /* (unbiased) round-to-even */ \
			((X + ((X >> (shift)) & 1) + (LLSHL1(shift-1)-1)) >> (shift))

			typedef signed long long SLL;
			if (show_data_bool) show_data("before scaling {");

			SLL acc = SLL(x);    // Assumed to be an integer already.
			acc *= unsigned(shift);
			x = double(LL_ROUND(acc,unsigned(shift)));

			if (show_data_bool) show_data("after scaling }");
			} break;
    }
	return x;
}
// SNPS EV rounding--->

}  // namespace detail

}  // namespace ruy
