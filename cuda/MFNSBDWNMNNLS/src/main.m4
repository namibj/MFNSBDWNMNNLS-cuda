#Copyright 2015 Merlin Kramer
#Licensed under the GNU Affero General Public License v3.0
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <assert.h>
#include <sm_30_intrinsics.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <driver_functions.h>
m4_define(`stop', `m4_dnl')
m4_changequote(`[', `]') stop ´´
m4_changequote([`], [´]
m4_define(`_DEF_concatVarSize´, eval(`1 ** 8´))
m4_define(`_DEF_m´, eval(`10 * 2´))m4_dnl The non-monotonic NNLS solvers inner iteration count M, which has to be even.
m4_define(`_DEF_N_target_optimization_F´, `1e-5´)m4_dnl The target value to get |\nabla f| to before stopping the optimizations of F
m4_define(`_DEF_N_target_optimization_X´, `1e-40´)m4_dnl The target value to get |\nabla f| to before stopping the optimizations of X
m4_define(`_DEF_BETA_F´, `0.5´)m4_dnl The non-monotonic NNLS solvers tweaking parameter β
m4_define(`_DEF_SIGMA_F´, `0.5´)m4_dnl The non-monotonic NNLS solvers tweaking parameter σ
m4_define(`_CALL_GEWICHTUNG´. `((1-abs(($1)- (0.5f * (_DEF_storedSizeX - 1))+0.5f)*(2.f/(_DEF_storedSizeX+1)))* (1-abs(($2)- (0.5f * (_DEF_storedSizeX - 1))+0.5f)*(2.f/(_DEF_storedSizeX+1))))´)m4_dnl The trusty macro to calculate the correct wheigh for a given coordinate
m4_define(`_DEF_BLOCk_TOO_HIGH_THREADS_XY´, `if (blockIdx.x == gridDim.x -1 && threadIdx.x >= (((_DEF_PATCH_NUM_X + 1) * _DEF_storedSizeX / 2 m4_ifelse(`Y´, `$1´, `2 * _DEF_SIZE_HALF_F´)) * ((_DEF_PATCH_NUM_Y + 1) * _DEF_storedSizeX / 2 m4_ifelse(`Y´, `$1´, `2 * _DEF_SIZE_HALF_F´)) - (gridDim.x -1) * blockDim.x) -1)´m4_dnl This is only the opening if(), not the {} nor an else
__global__ void kernel_set_float_zero(float* data, int lastBlockMax) {
	if (blockIdx.x != (gridDim.x - 1) || threadIdx.x < lastBlockMax)
		data[blockIdx.x * blockDim.x + threadIdx.x] = 0.f;

}

__global__ void  kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr(
		float* __restrict__ v_3, float* __restrict__ y, float* __restrict__ y_k,
		float* __restrict__ f_n_part_sums, float* __restrict__ f_n,
		unsigned int* __restrict__ count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float diff;
	float value;
	_DEF_BLOCK_TOO_HIGH_THREADS_XY(`Y´)
		diff = 0;
	else {
		diff = y[index] - y_k[index];
		v_3[index] = diff;
	}
	m4_dnl TODO: check if the reduction is suitable for macroisation and go on from there.
