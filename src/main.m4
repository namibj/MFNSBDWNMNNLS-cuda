/*
 *	Copyright 2015,2016 Merlin Kramer
 *	Licensed under the GNU Affero General Public License v3.0
 */
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <assert.h>
#include <sm_32_intrinsics.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <driver_functions.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui_c.h>
include(`m4_redefines.m4')dnl
define(`stop', defn(`dnl'))dnl
changequote(`[', `]') stop ´´)
changequote([`], [´])
include(`m4_changeword_evil_utf8.m4´)@dnl°
@define(`@stop´, `@dnl°´)@dnl°
@dnl° define(`@CALL_APPEND´,`define(`@DEF_MYLIST°´,ifdef(`@DEF_MYLIST´,`[changequote([,])@DEF_MYLIST°[,$1]changequote(`,´)]´,[$1]))´)
@define(`@DEF_NUM_PATCHES_X´, 5)@dnl°
@define(`@DEF_NUM_PATCHES_Y´, 5)@dnl°
@define(`@DEF_storedSizeX´, 128)@dnl°
@define(`@DEF_SLEEP_TIME_POLL´, 500)@dnl°
@define(`@DEF_FFT_SIZE´, @eval(2 ** 8))@dnl°
@define(`@DEF_SIZE_HALF_F´, @eval((@DEF_FFT_SIZE° - @DEF_storedSizeX°)/2))@dnl°
@define(`@DEF_SIZE_F´, @eval(@DEF_SIZE_HALF_F° * 2 + 1))@dnl°
@define(`@DEF_concatVarSize´, @eval(2 ** 8))@dnl°
@define(`@DEF_FFT_PRECISION´, `@ifelse(`R´, `$1´, `cufftReal´, `C´, `$1´, `cufftComplex´)´)@dnl°
@define(`@DEF_FFT_PRECISION_TYPE´, `float´)@dnl°
@define(`@DEF_m´, @eval(10 * 2))@dnl° The non-monotonic NNLS solvers inner iteration count M, which has to be even.
@define(`@DEF_N_target_optimization_F´, `1e-5´)@dnl° The target value to get |\nabla f| to before stopping the optimizations of F
@define(`@DEF_N_target_optimization_X´, `1e-40´)@dnl° The target value to get |\nabla f| to before stopping the optimizations of X
@define(`@DEF_BETA_F´, `0.5´)@dnl° The non-monotonic NNLS solvers tweaking parameter β for optimizeF()
@define(`@DEF_SIGMA_F´, `0.5´)@dnl° The non-monotonic NNLS solvers tweaking parameter σ for optimizeF()
@define(`@DEF_BETA_X´, `0.5´)@dnl° The non-monotonic NNLS solvers tweaking parameter β for optimizeX()
@define(`@DEF_SIGMA_X´, `0.5´)@dnl° The non-monotonic NNLS solvers tweaking parameter σ for optimizeX()
@define(`@CALL_GEWICHTUNG´, `((1-abs(($1)- (0.5f * (@DEF_storedSizeX° - 1))+0.5f)*(2.f/(@DEF_storedSizeX°+1)))* (1-abs(($2)- (0.5f * (@DEF_storedSizeX° - 1))+0.5f)*(2.f/(@DEF_storedSizeX°+1))))´)@dnl° The trusty macro to calculate the correct wheigh for a given coordinate
@define(`@DEF_BLOCK_TOO_HIGH_THREADS_XY´, `if (blockIdx.x == gridDim.x -1 && threadIdx.x >= (@eval(((@DEF_NUM_PATCHES_X° + 1) * @DEF_storedSizeX° / 2) * ((@DEF_NUM_PATCHES_Y° + 1) * @DEF_storedSizeX° / 2) -1) -(gridDim.x -1) * blockDim.x))´)@dnl° This is only the opening if(), not the {} nor an else
@define(`@DEF_NUM_PATCHES´, @eval(@DEF_NUM_PATCHES_X° * @DEF_NUM_PATCHES_Y°)) @dnl° just the total count of patches
@define(`@DEF_F_SQRD´, @eval(@DEF_SIZE_F° ** 2)) @dnl° the number of values in one f
@define(`@DEF_NUM_F_VALS´, @eval(@DEF_NUM_PATCHES° * @DEF_F_SQRD°)) @dnl° the total number of values of all f
@define(`@DEF_SIZE_Y´, @eval((@DEF_storedSizeX° / 2) ** 2 * (@DEF_NUM_PATCHES_X° + 1) * (@DEF_NUM_PATCHES_Y° + 1))) @dnl° the number of elements in an y-space

__global__ void kernel_set_float_zero(float* data, int lastBlockMax) {
	if (blockIdx.x != (gridDim.x - 1) || threadIdx.x < lastBlockMax)
		data[blockIdx.x * blockDim.x + threadIdx.x] = 0.f;

}
typedef int boolean;
struct store_f_X_T_1_informations {
	float* vec_f_o;
	float* vec_nabla_f_o;
	float alpha;
	float beta;
	float* nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d;
	float* nabla_f_scalar_prod_delta_f_part_sums;
	int block_num;
	int block_size;
	union{
	float* abs_vec_nabla_f_part_sums;
	float* abs_vec_delta_f_part_sums;
	};
};
struct streamCallback_informations{
	int b;
	float* f_n_h;
	boolean finished;
	struct store_f_X_T_1_informations * helper_struct_d;
	struct store_f_X_T_1_informations * helper_struct_h;
	float* part_sums_var_h;
	float* delta_nabla_f_part_sums_h;
	float* nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h;
	float f_o_h;
};
@define(`@CALL_BUTTERFLY_REDUCTION´,`{ for( int i = 16; i >= 1; i /= 2) $1 += __shfl_xor($1, i, 32);}´) @dnl° Basic additive XOR butterfly reduction across current warp on the given argument
@define(`@CALL_BUTTERFLY_BLOCK_REDUCTION´, `{ //Reduction
		@CALL_BUTTERFLY_REDUCTION($1)
		if(threadIdx.x%32==0) @ifelse(`s´, `$3´, `((float*) sharedPointer)[threadIdx.x / 32]´, `part_Sums[threadIdx.x / 32]´) = $1;
		__syncthreads();
		if(threadIdx.x/32 == 0) {
			$1 = @ifelse(`s´, `$3´, `(threadIdx.x/32 > blockDim.x/32)? 0 : ((float*) sharedPointer)[threadIdx.x & 0x1f];´, `part_Sums[threadIdx.x & 0x1f];´)
			@CALL_BUTTERFLY_REDUCTION($1)
			if (threadIdx.x%32==0) {$2}}}´) @dnl° whole 1024 Threads (in x index only) reduction across the block on $1, executing $2 at the end in the first thread of the block.
__global__ void  kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr(
		float* __restrict__ v_3, float* __restrict__ y, float* __restrict__ y_k,
		float* __restrict__ f_n_part_sums, float* __restrict__ f_n,
		unsigned int* __restrict__ count) {
	@define(`@DEF_conv_reduce´, `int index = blockIdx.x * blockDim.x + threadIdx.x;
	float diff;
	float value;
	__shared__ boolean isLastBlockDone;
	__shared__ float part_Sums[32];
	@DEF_BLOCK_TOO_HIGH_THREADS_XY°
		diff = 0;
	else {
		$1
	}
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`diff´, `$2[blockIdx.x] = diff;
		__threadfence();
		unsigned int value = atomicInc(count, gridDim.x);
		isLastBlockDone = (value == (gridDim.x - 1));´) @dnl° use that reduction!
	__syncthreads();
	if (isLastBlockDone) {
		if (gridDim.x >  blockDim.x) {
			value = 0;

			for (int x=0; (gridDim.x % blockDim.x) == 0 ? x < (gridDim.x / blockDim.x) : x <= (gridDim.x / blockDim.x); x++)
				value += (gridDim.x % blockDim.x) == 0 || threadIdx.x * blockDim.x < gridDim.x ? $2[threadIdx.x * blockDim.x] : 0;
		}
		@CALL_BUTTERFLY_BLOCK_REDUCTION(value, `*$3 = value;
		*count = 0;´) @dnl° reduction across the partial sums
	}´)
	@DEF_conv_reduce(`diff = y[index] - y_k[index];
		v_3[index] = diff;´, `f_n_part_sums´, `f_n´)
}

__global__ void kernel_nabla_tilde_Gets_nabla_capped_with_rule( float* __restrict__ vec_nabla_tilde_f, float* __restrict__ vec_nabla_f, float* __restrict__ vec_X) {
	@DEF_BLOCK_TOO_HIGH_THREADS_XY°
		return;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	vec_nabla_tilde_f[i] = (0 < vec_nabla_f[i] && 0 == vec_X[i]) ? 0 : vec_nabla_f[i];
}

__device__ @DEF_FFT_PRECISION(`R´) load_f_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ shared_Ptr) {
	@define(`@CALL_SPLIT_concatVar´, `int xPos = (offset / @DEF_concatVarSize°) & @DEF_concatVarSize°;
	int yPos = offset & @DEF_concatVarSize°;
	int patchNum = offset / @DEF_concatVarSize° / @DEF_concatVarSize°;´)
	@define(`@DEF_xPatchOffset´, @eval(((@DEF_storedSizeX° ** 2) * (@DEF_NUM_PATCHES_Y° + 1)) / 2))
	@define(`@DEF_yPatchOffset´, @eval(@DEF_storedSizeX° / 2))
	@define(`@CALL_RESTRICT_WITH_PADDING´, `@CALL_SPLIT_concatVar°
	@ifelse(`F´, `$1´, `int xPosStored, yPosStored;
	if (xPos <= @DEF_SIZE_HALF_F°) { @dnl° lower valid end
		xPosStored = xPos + @DEF_SIZE_HALF_F°;
	} else if (@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F° <= xPos) { @dnl° upper valid end
		xPosStored = xPos + @DEF_SIZE_HALF_F° - @DEF_FFT_SIZE°;
	} else {
		$2
	}
	if (yPos <= @DEF_SIZE_HALF_F°) {  @dnl° lower valid end
		yPosStored = yPos + @DEF_SIZE_HALF_F°;
	} else if (@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F° <= yPos) { @dnl° upper valid end
		yPosStored = yPos + @DEF_SIZE_HALF_F° - @DEF_FFT_SIZE°;
	} else {
		$2
	}´, `int zero_space[4]; @dnl° 0 <= x < 1 && 2 <= y < 3
	int xPatch = patchNum / @DEF_NUM_PATCHES_Y°;
	int yPatch = patchNum - xPatch * @DEF_NUM_PATCHES_Y°;
	if(0 == xPatch) { @dnl° x = 0 border
		zero_space[0] = @ifelse(`X´, `$1´, `@eval(@DEF_storedSizeX° / 2)´, `Y´, `$1´, `@eval(@DEF_SIZE_HALF_F° + (@DEF_storedSizeX° / 2))´);
		zero_space[1] = @ifelse(`s´, `$2´, `@DEF_FFT_SIZE°´, `l´, `$2´, `@eval(@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F°)´);
	} else if (@DEF_NUM_PATCHES_Y° - 1 == xPatch) { @dnl° x = MAX border
		zero_space[0] = @ifelse(`s´, `$2´, `0´, `l´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[1] = @ifelse(`s´, `$2´, `@DEF_FFT_SIZE°´, `l´, `$2´, `@eval(@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F°)´);
	} else { @dnl°  no x border
		zero_space[0] = @ifelse(`s´, `$2´, `0´, `l´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[1] = @ifelse(`X´, `$1´, `@eval(@DEF_FFT_SIZE°- (@DEF_storedSizeX° / 2))´, `Y´, `$1´, `@eval(@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F° - (@DEF_storedSizeX° / 2))´);
	}

	if (0 == yPatch) { @dnl° y = 0 border
		zero_space[2] = @ifelse(`X´, `$1´, `@eval(@DEF_storedSizeX° / 2)´, `Y´, `$1´, `@eval(@DEF_SIZE_HALF_F° + (@DEF_storedSizeX° / 2))´);
		zero_space[3] = @ifelse(`s´, `$2´, `@DEF_FFT_SIZE°´, `l´, `$2´, `@eval(@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F°)´);
	} else if (@DEF_NUM_PATCHES_X° - 1 == yPatch) { @dnl° y = MAX border
		zero_space[2] = @ifelse(`s´, `$2´, `0´, `l´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[3] = @ifelse(`s´, `$2´, `@DEF_FFT_SIZE°´, `l´, `$2´, `@eval(@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F°)´);
	} else { @dnl° no  y border
		zero_space[2] = @ifelse(`s´, `$2´, `0´, `l´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[3] = @ifelse(`X´, `$1´, `@eval(@DEF_FFT_SIZE°- (@DEF_storedSizeX° / 2))´, `Y´, `$1´, `@eval(@DEF_FFT_SIZE° - @DEF_SIZE_HALF_F° - (@DEF_storedSizeX° / 2))´);
	}

	int patchOffset = @DEF_xPatchOffset° * xPatch + @DEF_yPatchOffset° * yPatch;
	if (zero_space[0] <= xPos && xPos < zero_space[1] && zero_space[2] <= yPos && yPos < zero_space[3]) {
		int yPosStored = yPos - @DEF_SIZE_HALF_F°;
		int xPosStored = xPos - @DEF_SIZE_HALF_F°;
		$3
	}
	´)´) @dnl° args:
@dnl° $1 = 'F',
@dnl° 	uses:
@dnl° 		$2 = <statement('s) to execute in case this spot requires padding, it will be evaluated once in the x-coordinate checking, and once for the y-coordinate checking, in case it happens to be out of bounds in both.>
@dnl° 	sets (among others):
@dnl° 		xPos,yPos = <position in the fft, zero-indexed>,
@dnl° 		xPosStored,yPosStored = <position in memory, zero indexed>
@dnl° 		patchNum = <patch number, zero indexed>
@dnl° |$1 = 'Y',
@dnl° 	uses:
@dnl° 		$2 =
@dnl° 			'l' <to use the padding for loading a 'Y'>
@dnl° 			|'s' <to use the padding for saving a 'Y'>,
@dnl° 		$3 = <statement(s) to execute in case it is a valid position>
@dnl° |$1 = 'X',
@dnl° 	uses:
@dnl° 		$2 =
@dnl° 			'l' <to use the padding for loading a 'X'>
@dnl° 			|'s' <to use the padding for saving a 'X'>,
@dnl° 		$3 = <statement(s) to execute in case it is a valid position>
@CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
	return ((@DEF_FFT_PRECISION(`R´)*) dataIn)[(@DEF_SIZE_F° * @DEF_SIZE_F°) *  patchNum + @DEF_SIZE_F° * xPosStored + yPosStored] * @CALL_GEWICHTUNG(`xPos - @DEF_SIZE_HALF_F° - (@DEF_storedSizeX°/2)´, `yPos - @DEF_SIZE_HALF_F° - (@DEF_storedSizeX°/2)´);
}

__device__ void store_f_T_p_conj_fft_X(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = @ifelse(`float´, @DEF_FFT_PRECISION_TYPE°, `cuConjf´, `double´, @DEF_FFT_PRECISION_TYPE°, `cuConj´)(element); @dnl° TODO: insert right cmmand/
}

__device__ @DEF_FFT_PRECISION(`C´) load_F_X_m_F_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	return @ifelse(`float´, @DEF_FFT_PRECISION_TYPE°, `cuCmulf´)(((@DEF_FFT_PRECISION(`C´)*) (dataIn))[offset], ((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);  @dnl° TODO: include double precision as an option for the commplex multipliction.
}

__device__ void store_x_plus_x_weights_X(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	@CALL_RESTRICT_WITH_PADDING(`X´, `s´, `element *= .5f * @CALL_GEWICHTUNG(`xPosStored - (@DEF_storedSizeX° /2)´, `yPosStored - (@DEF_storedSizeX° /2)´);
	atomicAdd(&((@DEF_FFT_PRECISION(`R´)*) (dataOut))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored], element);
´)
}

__device__ @DEF_FFT_PRECISION(`R´) load_x_p_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	@DEF_FFT_PRECISION(`R´) ret;
	@CALL_RESTRICT_WITH_PADDING(`X´, `l´, `ret = ((@DEF_FFT_PRECISION(`R´)*) dataIn)[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored];
		ret *= @CALL_GEWICHTUNG(`xPosStored´, `yPosStored´);
´) else
		ret = 0;
	return ret;
}

__device__ @DEF_FFT_PRECISION(`R´) load_f_X_1_l_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	@CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
	return ((@DEF_FFT_PRECISION(`R´)*) dataIn)[(@DEF_SIZE_F° * @DEF_SIZE_F°) * patchNum + @DEF_SIZE_F° * xPosStored + yPosStored];
}

__device__ @DEF_FFT_PRECISION(`R´) load_v_3_X_T_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	@CALL_RESTRICT_WITH_PADDING(`Y´, `l´, `return ((@DEF_FFT_PRECISION(`R´)*) (dataIn))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored];´) else
		return 0;
}

__device__ void store_f_X_y_p_v_1_F(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	@CALL_RESTRICT_WITH_PADDING(`Y´, `s´, `atomicAdd(&((@DEF_FFT_PRECISION(`R´)*) (dataOut))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored], element * ((float) (1. / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE°))));´)
}

__device__ void store_f_X_fft_m_x_F(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = cuCmulf(((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset], ((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);
}

__device__ void store_f_X_T_fft_m_x_F(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = cuCmulf(((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset], cuConjf(((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]));
}
@define(`@DEF_STORE_REDUCE_CALL´, `@ifelse(`11´, `$1´, `store_f_X_T_1_nabla_tilde_f_uneven_b_F´, `12´, `$1´, `store_f_X_T_1_nabla_tilde_f_even_b_F´, `21´, `$1´, `store_f_X_T_2_delta_tilde_f_even_b_F´, `22´, `$1´, `store_f_X_T_2_delta_tilde_f_uneven_b_F´)´)
@define(`@DEF_STORE_REDUCE_DEF´, `__device__ void @DEF_STORE_REDUCE_CALL(`$1$2´) (void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	struct store_f_X_T_1_informations (*inform_struct) =
			(store_f_X_T_1_informations*) (callerInfo);
	float nabla_tilde_f = 0;
	float value;
	float f;
	float *vec_f_o = inform_struct->vec_f_o;
	float *vec_f = vec_f_o;
	float *vec_nabla_tilde_f_o = inform_struct->vec_nabla_f_o;
	bool isF = true;
	@CALL_RESTRICT_WITH_PADDING(`F´, `value = 0;
		isF = false;
		goto sumItUp;´)
	int index = (@DEF_SIZE_F° * @DEF_SIZE_F°) * patchNum + @DEF_SIZE_F° * xPosStored + yPosStored;
	@ifelse(`1´, `$1´, `if (element > 0 &&  0 == vec_f_o[index])
		nabla_tilde_f = 0;
	else
		nabla_tilde_f = ((float) .5 / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE°)) * element,

	f = vec_f_o[index] - inform_struct->alpha * inform_struct->beta * nabla_tilde_f;
	value = vec_nabla_tilde_f_o[index] * (vec_f_o[index] - vec_f[index]);´, `value = inform_struct->vec_nabla_f_o[index] * element * ((float) 1. / (@DEF_FFT_SIZE° *  @DEF_FFT_SIZE°));´)

	sumItUp:

	@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, `@ifelse(`1´, `$1´, `inform_struct->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d´, `inform_struct->nabla_f_scalar_prod_delta_f_part_sums´)[gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockDim.y + blockIdx.x] = value;
	inform_struct->block_num = gridDim.x * gridDim.y * gridDim.z;
		inform_struct->block_size = blockDim.x * blockDim.y * blockDim.z;´, `s´)

	value = @ifelse(`1´, `$1´, `nabla_tilde_f * nabla_tilde_f;´, `element * element * ((float) 1. / (@DEF_FFT_SIZE°l * @DEF_FFT_SIZE° * @DEF_FFT_SIZE° * @DEF_FFT_SIZE°));´)
	@ifelse(`2´, `$2´, `@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, 	`@ifelse(`1´, `$1´, `inform_struct->abs_vec_nabla_f_part_sums´, `inform_struct->abs_vec_delta_f_part_sums´)[gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x] = value;´, `s´)´)

	if (isF)
		vec_f_o[index] = f;
}´)

@DEF_STORE_REDUCE_DEF°(1, 2)

@DEF_STORE_REDUCE_DEF°(1, 1)

@DEF_STORE_REDUCE_DEF°(2, 1)

@DEF_STORE_REDUCE_DEF°(2, 2)

__device__ @DEF_FFT_PRECISION(`R´) load_x_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	@DEF_FFT_PRECISION(`R´) ret;
	@CALL_RESTRICT_WITH_PADDING(`X´, `l´, `ret = ((@DEF_FFT_PRECISION(`R´)*) dataIn)[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored];
´) else
		ret = 0;
	return ret;
}
__device__ @DEF_FFT_PRECISION(`C´) load_x_p_cmplx_mul_f_p(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	return ((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = cuCmulf(((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset], ((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);
} @dnl° convert the parameters and rest from store to load.

__device__ void store_y_plus_y_X(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	@CALL_RESTRICT_WITH_PADDING(`Y´, `s´, `atomicAdd(&((@DEF_FFT_PRECISION(`R´)*) (dataOut))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored], element * ((float) (1. / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE°))));´)
}

@dnl° that have been all the function definitions for the device side, except the not refactored, but to be coded, device side reduction/summation code for those reductions previously done in host code (to seriously reduce host<.device traffic
@define(`@COMPOUND´, `@ifelse(`L´, $1, `load´, `S´, `$1´, `store´)_$3´)
@define(`@CALL_ALLOC_CB´, `__device__ cufftCallback@ifelse(`L´, `$1´, `Load´, `S´, `$1´, `Store´)$2 _d_@COMPOUND°($@) = @COMPOUND($@);
cufftCallback@ifelse(`L´, `$1´, `Load´, `S´, `$1´, `Store´)$2 _h_@COMPOUND($@);@divert(1)	cudaMemcpyFromSymbol(&_h_@COMPOUND($@), _d_@COMPOUND($@), sizeof(_h_@COMPOUND($@)));
@divert(0)´) @stop° ´)´)
@dnl° TODO: insert all the @CALL_ALLOC_CB
@CALL_ALLOC_CB(`L´, `R´, `x_p_F´)
@CALL_ALLOC_CB(`L´, `R´, `f_X_1_l_F´)
@CALL_ALLOC_CB(`S´, `C´, `f_X_fft_m_x_F´)
@CALL_ALLOC_CB(`S´, `R´, `f_X_y_p_v_1_F´)
@CALL_ALLOC_CB(`S´, `C´, `f_X_T_fft_m_x_F´)
@CALL_ALLOC_CB(`L´, `R´, `v_3_X_T_F´)
@CALL_ALLOC_CB(`S´, `R´, `f_X_T_1_nabla_tilde_f_even_b_F´)
@CALL_ALLOC_CB(`S´, `R´, `f_X_T_1_nabla_tilde_f_uneven_b_F´)
@CALL_ALLOC_CB(`S´, `R´, `f_X_T_2_delta_tilde_f_even_b_F´)
@CALL_ALLOC_CB(`S´, `R´, `f_X_T_2_delta_tilde_f_uneven_b_F´)
@CALL_ALLOC_CB(`L´, `R´, `f_p_X´)
@CALL_ALLOC_CB(`S´, `C´, `f_T_p_conj_fft_X´)
@CALL_ALLOC_CB(`L´, `R´, `x_p_X´)
@CALL_ALLOC_CB(`L´, `C´, `F_X_m_F_X´)
@CALL_ALLOC_CB(`S´, `R´, `y_plus_y_X´)
@CALL_ALLOC_CB(`S´, `R´, `x_plus_x_weights_X´)
void getCallbacks() {
	@undivert(1)}
@define(`@CALL_ROUND_BLOCK_SIZE_UP´, `((($1) % ($2) ? ($1) / ($2) : ($1) / ($2) + 1))´)
int setFloatDeviceZero(float* data, size_t count, int blocksize, cudaStream_t stream) {
	kernel_set_float_zero<<<@CALL_ROUND_BLOCK_SIZE_UP(`count´, `blocksize´), blocksize, 0, stream>>>(data, count % blocksize);
	return 0;
}

__global__ void __launch_bounds__(1024, 1) optimizeFcallback(struct streamCallback_informations * __restrict__ informations) {

	__shared__ float part_Sums[32];

	float value;

	__shared__ float delta_nabla_f;
	__shared__ float delta_or_nabla_abs;
	__shared__ float complicatedSums;

	float * delta_nabla_f_part_sums = informations->helper_struct_d->nabla_f_scalar_prod_delta_f_part_sums;
	float * part_sums_var_d = informations->helper_struct_d->abs_vec_nabla_f_part_sums;
	float * nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d = informations->helper_struct_d->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d;

	value = 0;
	//for (informations->helper_struct_d->block_size / 1024)
		value = delta_nabla_f_part_sums[threadIdx.x];
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, `delta_nabla_f = value;´)

	value = 0;
	//for (informations->helper_struct_d->block_size / 1024)
		value = part_sums_var_d[threadIdx.x];
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, `delta_or_nabla_abs = value;´)

	value = 0;
	//for (informations->helper_struct_d->block_size / 1024)
		value = nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d[threadIdx.x];
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, `complicatedSums = value;´)

	if (threadIdx.x == 0) {
		if (informations->b % 2 == 0) {
			if (*(informations->f_n_h) < @DEF_N_target_optimization_F°) {
				@dnl° optimization is finished
				informations->finished = 2;
				return;
			} else
				informations->finished = 1;
			informations->helper_struct_h->alpha = delta_or_nabla_abs / delta_nabla_f;
		} else {
			informations->helper_struct_h->alpha = delta_nabla_f / delta_or_nabla_abs;
		}
		if (informations->f_o_h - *(informations->f_n_h) <= @DEF_SIGMA_F° * complicatedSums)
			informations->helper_struct_d->beta *= @DEF_BETA_F°;
		informations->f_o_h = *(informations->f_n_h);
	}
}

int optimizeF(float* f_h, float* x_h, float* y_k_h, cudaStream_t stream) {
	@define(`@_free_stack´, `@ifdef(`@_free_stack1´, `@_free_stack1°@popdef(`@_free_stack1´)@_free_stack°´)´)
	@define(`@DEF_CU_MALLOC´, `@ifelse(`ndef´, `$5´,,`$2* ´)$1 = NULL;
	cudaMalloc´@ifelse(`h´, `$4´, `Host@pushdef(`@_free_stack1´, `cudaFreeHost($1);
´)´, `@pushdef(`@_free_stack1´, `$1´)´)`((void**) &$1, sizeof($2) * $3);´) @dnl° $1 = [device] pointer name, $2 = [device] pointer type (without the '*'), $3 = number of elements to allocate[, $4 = h (to allocate host space)
	@define(`@DEF_CU_MALLOC_HTDC´, `@DEF_CU_MALLOC($@)@ifelse(`´, `$7´,,`
$7´)@divert(1)
	cudaMemcpyAsync($1, $5, sizeof($2) * $3, cudaMemcpyHostToDevice, $6);@divert(0)@ifelse(`´, `$8´,,`@divert(2)
$8@divert(0)´)´) @dnl° $1 = device pointer name, $2 = device pointer type (without the '*'), $3 = number of elements to allocate, $4 = '' (just jump with a double ','), $5 = host pointer name, $6 = stream, $7 = optional (somthing to execute after the allocation and before scheduling the copy for the bunch of copys, $8 = optional (to execute after copying)
	@DEF_CU_MALLOC_HTDC(`f_d´, `float´, `@DEF_NUM_F_VALS°´,, `f_h´, `stream´)
	@DEF_CU_MALLOC_HTDC(`y_k_d´, `float´, `@DEF_SIZE_Y°´,, `y_k_h´, `stream´)
	@DEF_CU_MALLOC_HTDC(`helper_struct_d´, `store_f_X_T_1_informations´, 1,, `helper_struct_h´, `stream´, `@DEF_CU_MALLOC(`helper_struct_h´, `store_f_X_T_1_informations´, 1, `h´)
	@DEF_CU_MALLOC(`streamCallback´, `struct streamCallback_informations´, `1´, `h´)
	helper_struct_h->alpha = 0.5;
	helper_struct_h->beta = 0.5;´)
	@DEF_CU_MALLOC(`f_n_h´, `float´, `1´, `h´)
	@DEF_CU_MALLOC(`helper_struct_h->vec_f_o´, `float´, `@DEF_NUM_F_VALS°´,,`ndef´)
	@DEF_CU_MALLOC(`helper_struct_h->vec_nabla_f_o´, `float´, `@DEF_NUM_F_VALS°´,,`ndef´)
	@DEF_CU_MALLOC(`x_p_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°))
	@DEF_CU_MALLOC_HTDC(`v_3_d´, `float´, `@DEF_SIZE_Y°´,,`x_h´, `stream´,,`cufftExecR2C(plan_x_p_F, v_3_d, x_p_d);
	setFloatDeviceZero(v_3_d, @DEF_SIZE_Y°, 128, stream);´)
	@DEF_CU_MALLOC(`y_d´, `float´, `@DEF_SIZE_Y°´)
	@DEF_CU_MALLOC(`v_tmp_cmplx_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°))
	@DEF_CU_MALLOC(`f_n_d´, `float´, 1)
	@DEF_CU_MALLOC(`count_d´, `unsigned int´, 1)
	@DEF_CU_MALLOC(`f_n_part_sums_d´, `float´, @CALL_ROUND_BLOCK_SIZE_UP(@DEF_SIZE_Y°, 1024))
	@DEF_CU_MALLOC(`helper_struct_h->abs_vec_nabla_f_part_sums´, `float´, @DEF_NUM_F_VALS°,,`ndef´)
	@DEF_CU_MALLOC(`helper_struct_h->abs_vec_delta_f_part_sums´, `float´, @DEF_NUM_F_VALS°,,`ndef´)
	@DEF_CU_MALLOC(`helper_struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d´, `float´, @DEF_NUM_F_VALS°,,`ndef´)
	@DEF_CU_MALLOC(`helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums´, `float´, 32768,,`ndef´) @dnl° TODO: care and decide about helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums size '32768'
	(*streamCallback).finished = 0;
	(*streamCallback).helper_struct_d = helper_struct_d;
	(*streamCallback).helper_struct_h = helper_struct_h;
	(*streamCallback).f_n_h = f_n_h;
	@dnl° TODO: check for memset to f_n_d (if it is necessary)
	@define(`@nargs´, `$#´) @dnl° just emit the number of arguments given. Usefull to determine the size of a grouped argument.
	@define(`@_echo_q´, `$@´) @dnl° just a macro to ecpand into all the args, qouted. Usefull to expand a grouped argument.
	@define(`@echo_1´, ``$1´´)
	@define(`@echo_2´, ``$2´´)
	@define(`@_CB_PLAN_STMT´, `@_CB_PLAN_STMT1(`$4´, `$2´, `$3´, (`$#´, `$5´), @_echo_q°$1)´)
	@define(`@_CB_PLAN_STMT1´, `@ifelse(5, @echo_1$4, `for (int k = 0; k < @echo_2$4; k++) ´)cufftXtSetCallback(plan_$2@ifelse(5, @echo_1$4, `[k]´), ((void**) &_h_@ifelse(`l´, `$1´, `load_´, `s´, `$1´, `store_´)@ifelse(`´, `$5´, `$2´, $5)), CUFFT_CB_@ifelse(`lC´, `$1$3´, `LD_COMPLEX´, `lR´, `$1$3´, `LD_REAL´, `sC´, `$1$3´, `ST_REAL´, `sR´, `$1$3´, `ST_COMPLEX´), @ifelse(`´, `$6´, `NULL´, `((void**) &$6_d@ifelse(5, @echo_1$4, `[k]´))´));@ifelse(`6´, `$#´, `´, `
		cufftXtSetCallbackSharedSize(plan_$2@ifelse(5, @echo_1$4, `[k]´), CUFFT_CB_@ifelse(`lC´, `$1$3´, `LD_COMPLEX´, `lR´, `$1$3´, `LD_REAL´, `sC´, `$1$3´, `ST_REAL´, `sR´, `$1$3´, `ST_COMPLEX´), $7);´)´)
		@dnl° TODO: insert the shared memory reservation call (with semicolon)
	@define(`@DEF_CUFFT_HANDLE´, `cufftHandle plan_$1@ifelse(6, `$#´, `[$6]´);
	{
		@ifelse(6, `$#´, `for(int k = 0; k < $6; k++) cufftCreate(&plan_$1[k]);´, `cufftCreate(&plan_$1);´)
		int inembed[] = { 1, @ifelse(`C´, `$2´, @eval(@DEF_FFT_SIZE° / 2 + 1), @DEF_concatVarSize°) };
		int onembed[] = { 1, @ifelse(`C´, `$2´, @DEF_concatVarSize°, @eval(@DEF_FFT_SIZE° / 2 + 1)) };
		int n[] = { @DEF_FFT_SIZE°, @DEF_FFT_SIZE° };
		@ifelse(6, `$#´, `for(int k = 0; k < $6; k++) ´)cufftPlanMany(@ifelse(6, `$#´, `&(plan_$1[k])´, `&plan_$1´), 2, n, inembed, 1, @ifelse(`C´, `$2´, `@eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1))´, `R´, `$2´, `@eval(@DEF_concatVarSize° ** 2)´), onembed, 1, @ifelse(`C´, `$2´, `@eval(@DEF_concatVarSize° ** 2), CUFFT_C2R´, `R´, `$2´, `@eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1)), CUFFT_R2C´), @DEF_NUM_PATCHES°);
		@ifelse(6, `$#´, `for(int k = 0; k < $6; k++) cufftSetStream(plan_$1[k], $3[k]);´, `cufftSetStream(plan_$1, $3);´)
		@ifelse(@nargs$4, `1´, `´, `@_CB_PLAN_STMT($4, `$1´, `$2´, `l´@ifelse(6, `$#´, `, `$6´´))´)
		@ifelse(@nargs$5, `1´, `´, `@_CB_PLAN_STMT($5, `$1´, `$2´, `s´@ifelse(6, `$#´, `, `$6´´))´)
	}´)
	@dnl° $1 = name of the plan, without the leading plan_, $2 = 'C' if C2R; 'R' if R2C, $3 = name of the stream to execute in, $4 = ([[loadCallbackName <without the leading _h_load_, if omitted: _h_load_$1>], [callerInfo device pointer<without the trailing _d, if omitted: NULL>] <to omit: leave the parenthesis empty and omit the comma in between>]), \
	@dnl° $5 = ([[storeCallbackName <without the leading _h_store_, if omitted: _h_store_$1>], [callerInfo device pointer <without the trailing _d, if omited: NULL>][, size to request for __shared__ memory allocation <inclusive any sizeof(...) factors>]<to omit: leave the parenthesis empty and omit the comma in between>])
	@dnl° [$6 = num_plan_clones_with_streams]
	@undivert(1)
	@DEF_CUFFT_HANDLE°(`x_p_F´, `R´, `stream´, (,), ())
	@DEF_CUFFT_HANDLE°(`f_X_1_l_F´, `R´, `stream´, (,), (`f_X_fft_m_x_F´, `x_p´))
	@DEF_CUFFT_HANDLE°(`f_X_1_s_F´, `C´, `stream´, (), (`f_X_y_p_v_1_F´,))
	@DEF_CUFFT_HANDLE°(`f_X_T_l_F´, `R´, `stream´, (`v_3_X_T_F´,), (`f_X_T_fft_m_x_F´, `x_p´))
	@DEF_CUFFT_HANDLE°(`f_X_T_1_nabla_tilde_f_even_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@DEF_CUFFT_HANDLE°(`f_X_T_1_nabla_tilde_f_uneven_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@DEF_CUFFT_HANDLE°(`f_X_T_2_delta_tilde_f_even_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@DEF_CUFFT_HANDLE°(`f_X_T_2_delta_tilde_f_uneven_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@undivert(2)
	do {
		for (int b=0; b < @DEF_m°; b++) {
			cufftExecR2C(plan_f_X_1_l_F, f_d, v_tmp_cmplx_d);
			setFloatDeviceZero(y_d, @DEF_SIZE_Y°, 128, stream);
			cufftExecC2R(plan_f_X_1_s_F, v_tmp_cmplx_d, y_d);
			kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr<<<@CALL_ROUND_BLOCK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´), 1024, 0, stream>>>(v_3_d, y_d, y_k_d, f_n_part_sums_d, f_n_d, count_d);
			cufftExecR2C(plan_f_X_T_l_F, v_3_d, v_tmp_cmplx_d);
			if(b % 2 == 0)
				cufftExecC2R(plan_f_X_T_1_nabla_tilde_f_even_b_F, v_tmp_cmplx_d, v_3_d);
			else
				cufftExecC2R(plan_f_X_T_1_nabla_tilde_f_uneven_b_F, v_tmp_cmplx_d, v_3_d);
			cufftExecR2C(plan_f_X_1_l_F, helper_struct_h->vec_nabla_f_o, v_tmp_cmplx_d);
			setFloatDeviceZero(y_d, @DEF_SIZE_Y°, 128, stream);
			cufftExecC2R(plan_f_X_1_s_F, v_tmp_cmplx_d, v_3_d);
			cufftExecR2C(plan_f_X_T_l_F, v_3_d, v_tmp_cmplx_d);
			if(b % 2 == 0)
				cufftExecC2R(plan_f_X_T_2_delta_tilde_f_even_b_F, v_tmp_cmplx_d, v_3_d);
			else
				cufftExecC2R(plan_f_X_T_2_delta_tilde_f_uneven_b_F, v_tmp_cmplx_d, v_3_d);
			cudaMemcpyAsync((void*) helper_struct_h, (void*) helper_struct_d, sizeof(store_f_X_T_1_informations), cudaMemcpyDeviceToHost, stream);
			//cudaMemcpyAsync((void*) delta_nabla_f_part_sums_h, (void*) helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			//if (b % 2 == 0)
			//	cudaMemcpyAsync((void*) part_sums_var_h, (void*) helper_struct_h->abs_vec_nabla_f_part_sums, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			//else
			//	cudaMemcpyAsync((void*) part_sums_var_h, (void*) helper_struct_h->abs_vec_delta_f_part_sums, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			cudaMemcpyAsync((void*) f_n_h, (void*) f_n_d, sizeof(float), cudaMemcpyDeviceToHost, stream);
			//cudaMemcpyAsync((void*) (*streamCallback).nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h, (void*) helper_struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			optimizeFcallback<<<1, 1024, 0, stream>>>(streamCallback);
			if (b % 2 == 0) {
				while((*streamCallback).finished == 0)
					usleep(@DEF_SLEEP_TIME_POLL°);
				if ((*streamCallback).finished == 2) @dnl° 2 means precision reached, 1 means precision not reached, i.e. schedule the next iteration, 0 means noz yet decided.
					goto end_loop;
				else
					(*streamCallback).finished = 0; @dnl° reset the poll flag if used again.
			}
			cudaMemcpyAsync(helper_struct_d, helper_struct_h, sizeof(store_f_X_T_1_informations), cudaMemcpyHostToDevice, stream);
		}
	} while (true); @dnl° TODO: check if while (true) is really the right thing to do here.
	end_loop: cudaMemcpyAsync((void*) f_h, (void*) f_d, sizeof(float) * @DEF_NUM_F_VALS°, cudaMemcpyDeviceToHost, stream);
	while(cudaErrorNotReady == cudaStreamQuery(stream))
		usleep(@DEF_SLEEP_TIME_POLL°);
	@_free_stack°
	return 0;
}
@dnl° __device__ @DEF_FFT_PRECISION(`R´) load_f_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
@dnl°	@CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
@dnl°	return @CALL_GEWICHTUNG(`xPos´, `yPos´) * ((@DEF_FFT_PRECISION(`R´)*) dataIn)[(@DEF_SIZE_F° * @DEF_SIZE_F°) * patchNum + SIZE_F * xPosStoed + yPosStored];
@dnl°} @dnl° TODO: check why this is already done above. seems kinda strange, but it may be the first part of optimizeX() I did back then... also this is probably not the right call to @CALL_GEWICHTUNG(), as I did it differently above.


__global__ void kernel_nabla_f_to_nabla_tilde_f_X(const float* const __restrict__ v_4, float* __restrict__ X, float* __restrict__ nabla_tilde_f, const float * const __restrict__ alpha_beta, float* __restrict__ scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc, unsigned int* __restrict__ count, float* __restrict__ thread_part_sums) {
	@DEF_conv_reduce(`float x_o_val = X[index];
		float v_4_val = v_4[index] * .5f;
		float nabla_f_o_val = nabla_tilde_f[index];
		float nabla_tilde_val = v_4_val > 0 && 0 == x_o_val ? 0 : v_4_val;
		float x_val = x_o_val - (*alpha_beta) * nabla_tilde_val;
		diff = nabla_f_o_val * (x_o_val - x_val);
		nabla_tilde_f[index] = nabla_tilde_val;
		X[index] = x_val;´, `thread_part_sums´, `scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc´)
}
__global__ void kernel_delta_nabla_tilde_f_X(float3* const __restrict__ thread_part_sums, const float* const __restrict__ nabla_tilde_f, const float* const __restrict__ delta_tilde_f, const float* const __restrict__ f_n, double* const __restrict__ beta, const int* const __restrict__ b, unsigned int* const __restrict__ count, const float* const __restrict__ scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc, float* const __restrict__ f_o, float* const __restrict__ a, float* __restrict__ alpha_beta, const int num_images, boolean * const __restrict__ finished) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float f_n_i;
	float scalar_prod;
	float abs;
	__shared__ boolean isLastBlockDone;
	__shared__ float part_Sums[32];
	@DEF_BLOCK_TOO_HIGH_THREADS_XY°{
		scalar_prod = 0;
		abs = 0;
	} else {
		scalar_prod = nabla_tilde_f[index] * delta_tilde_f[index];
		abs = (*b%2==0?nabla_tilde_f:delta_tilde_f)[index];
	}
	if(index >= num_images)
		f_n_i = 0;
	else
		f_n_i = f_n[index];
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`abs´, `thread_part_sums[blockIdx.x].x = abs;´)
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`scalar_prod´, `thread_part_sums[blockIdx.x].y = scalar_prod;´)
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`f_n_i´, `thread_part_sums[blockIdx.x].z = f_n_i;
		__threadfence();
		unsigned int value = atomicInc(count, gridDim.x);
		isLastBlockDone = (value == (gridDim.x - 1));´)
	__syncthreads();
	if (isLastBlockDone) {
		if (gridDim.x >  blockDim.x) {
			abs = 0;
			for (int x=0; (gridDim.x % blockDim.x) == 0 ? x < (gridDim.x / blockDim.x) : x <= (gridDim.x / blockDim.x); x++)
				abs += (gridDim.x % blockDim.x) == 0 || threadIdx.x * blockDim.x < gridDim.x ? thread_part_sums[threadIdx.x * blockDim.x].x : 0;
			scalar_prod = 0;
			for (int x=0; (gridDim.x % blockDim.x) == 0 ? x < (gridDim.x / blockDim.x) : x <= (gridDim.x / blockDim.x); x++)
				scalar_prod += (gridDim.x % blockDim.x) == 0 || threadIdx.x * blockDim.x < gridDim.x ? thread_part_sums[threadIdx.x * blockDim.x].y : 0;
			f_n_i = 0;
			for (int x=0; (gridDim.x % blockDim.x) == 0 ? x < (gridDim.x / blockDim.x) : x <= (gridDim.x / blockDim.x); x++)
				f_n_i += (gridDim.x % blockDim.x) == 0 || threadIdx.x * blockDim.x < gridDim.x ? thread_part_sums[threadIdx.x * blockDim.x].z : 0;
		}
		@CALL_BUTTERFLY_BLOCK_REDUCTION(`abs´, `´) @dnl° reduction across the partial sums
		@CALL_BUTTERFLY_BLOCK_REDUCTION(`scalar_prod´, `´) @dnl° reduction across the partial sums
		@CALL_BUTTERFLY_BLOCK_REDUCTION(`f_n_i´, `if(*b%2==0)
			if(abs<@DEF_N_target_optimization_X°) *finished = true;
			else *a = (float) (((double) abs) / ((double) scalar_prod));
		else  *a = (float) (((double) scalar_prod) / ((double) abs));
		if((*b%@DEF_m°) == (@DEF_m° - 1) && (*f_o - f_n_i) <= (@DEF_SIGMA_X° * (*scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc)))
			*beta *= @DEF_BETA_X°;
		*alpha_beta = *a * *beta;
		*f_o = f_n_i; @dnl° TODO: make sure to do via pointer-switching (double buffering): nabla_f_o = nabla_tilde_F
		*count = 0;´) @dnl° reduction across the partial sums
	}
}
@dnl° scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__b @dnl° figure out where this came from. Git can help.
void optimizeX(float** f_h, float* x_h, float** y_k_h, int num_images){ @dnl° TODO: convert the symbolic code to actual code
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	@DEF_CU_MALLOC(`finished´, `boolean´, 1, `h´)
	@DEF_CU_MALLOC(`alpha_beta´, `float´, 1, `h´)
	@DEF_CU_MALLOC(`b´, `int´, 1, `h´)
	@DEF_CU_MALLOC(`beta´, `double´, 1, `h´)
	@DEF_CU_MALLOC(`f_o´, `float´, 1, `h´)
	@DEF_CU_MALLOC(`a´, `float´, 1, `h´)
	@DEF_CU_MALLOC(`f_n´, `float´, num_images, `h´)
	@DEF_CU_MALLOC(`scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc´, `float´, 1, `h´)
	*finished = false;
	*alpha_beta = 1;
	*beta = 1;
	*a = 1;
	@DEF_CU_MALLOC(`count_d´, `unsigned int´, 1)
	@DEF_CU_MALLOC(`x_p_X_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°))
	@DEF_CU_MALLOC(`f_n_part_sums_d´, `float´, `@CALL_ROUND_BLOCK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´)´)
	@DEF_CU_MALLOC(`tri_part_sums´, `float3´, `@CALL_ROUND_BLOCK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´)´)
	@DEF_CU_MALLOC(`v_tmp_cmplx_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°))
	@DEF_CU_MALLOC(`f_p_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°))
	@DEF_CU_MALLOC(`f_t_p_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°))
	@DEF_CU_MALLOC(`nabla_tilde_f´, `float´, `@DEF_SIZE_Y°´)
	@DEF_CU_MALLOC(`v_4_d´, `float´, `@DEF_SIZE_Y°´)
	@DEF_CU_MALLOC(`v_3_d´, `float´, `@DEF_SIZE_Y°´)
	@DEF_CU_MALLOC(`x_d´, `float´, `@DEF_SIZE_Y°´)

	@DEF_CUFFT_HANDLE(`f_p_X´, `R´, `stream´, (,), ())
	@DEF_CUFFT_HANDLE(`f_p_X_T´, `R´, `stream´, (`f_X_1_l_F´,), (`f_T_p_conj_fft_X´,))
	@DEF_CUFFT_HANDLE(`x_p_X´, `R´, `stream´, (,), ())
	@DEF_CUFFT_HANDLE(`F_k´, `C´, `stream´, (`F_X_m_F_X´, `x_p_X´), (`y_plus_y_X´,))
	@DEF_CUFFT_HANDLE(`v_3_X_T_F´, `R´, `stream´, (,), ())
	@DEF_CUFFT_HANDLE(`F_T_k´, `C´, `stream´, (`F_X_m_F_X´, `v_tmp_cmplx´), (`x_plus_x_weights_X´,))

	while(!(*finished)){ @dnl° fix this. Could result in an infinite loop if machine precision limits are reached in a bad enough way.
		for (*b = 0; *b < @DEF_m°; (*b)++) {

			setFloatDeviceZero(v_4_d, @DEF_SIZE_Y°, 128, stream);

			cufftExecR2C(plan_x_p_X, x_d, x_p_X_d);

			for (int k = 0; k < num_images; k++) {
				cufftExecR2C(plan_f_p_X, f_h[k], f_p_d);
				cufftExecR2C(plan_f_p_X_T, f_h[k], f_t_p_d);

				setFloatDeviceZero(v_3_d, @DEF_SIZE_Y°, 128, stream);

				cufftExecC2R(plan_F_k, f_p_d, v_3_d);

				kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr<<<@CALL_ROUND_BLOCK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´), 1024, 0, stream>>>(v_3_d, v_3_d, y_k_h[k], f_n_part_sums_d, &f_n[k], count_d);
				cufftExecR2C(plan_v_3_X_T_F, v_3_d, v_tmp_cmplx_d);
				cufftExecC2R(plan_F_T_k, f_t_p_d, v_4_d);
			}

			kernel_nabla_f_to_nabla_tilde_f_X<<<@CALL_ROUND_BLOCK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´), 1024, 0, stream>>>(v_4_d, x_d, nabla_tilde_f, alpha_beta, scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc, count_d, f_n_part_sums_d);
			cufftExecR2C(plan_x_p_X, nabla_tilde_f, x_p_X_d);

			setFloatDeviceZero(v_4_d, @DEF_SIZE_Y°, 128, stream);

			for (int k = 0; k < num_images; k++) {
				cufftExecR2C(plan_f_p_X, f_h[k], f_p_d);
				cufftExecR2C(plan_f_p_X_T, f_h[k], f_t_p_d);

				setFloatDeviceZero(v_3_d, @DEF_SIZE_Y°, 128, stream);

				cufftExecC2R(plan_F_k, f_p_d, v_3_d);@dnl° TODO: check this out. Create an Issue for it.

				cufftExecR2C(plan_v_3_X_T_F, v_3_d, v_tmp_cmplx_d);
				cufftExecC2R(plan_F_T_k, f_t_p_d, v_4_d);
			}

			kernel_delta_nabla_tilde_f_X<<<@CALL_ROUND_BLOCK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´), 1024, 0, stream>>>(tri_part_sums, nabla_tilde_f, v_4_d, f_n, beta, b, count_d, scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc, f_o, a, alpha_beta, num_images, finished);
		}
	}
	while(cudaErrorNotReady == cudaStreamQuery(stream)) sched_yield();
	cudaStreamDestroy(stream);
}
typedef struct optimizeF_helper_struct {
	float* f_h;
	float* y_k_h;
	float* x_h; @dnl° temporary to fix more typos.
} optimizeF_helper_struct_t;
void optimizeF_helper(optimizeF_helper_struct_t data) {
	//optimizeF_helper_struct_t *data = (optimizeF_helper_struct_t*) datav;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	optimizeF(data.f_h, data.y_k_h, data.x_h, stream);
	while(cudaErrorNotReady == cudaStreamQuery(stream)) sched_yield();
	cudaStreamDestroy(stream);
	return;
}
void computeRecursive(float** f_h, float** y_k_h, float* x, int num_images){
		if (num_images > 2) {
			@DEF_CU_MALLOC(`x_h_1´, `float´, `@DEF_SIZE_Y°´, `h´)
			@DEF_CU_MALLOC(`x_h_2´, `float´, `@DEF_SIZE_Y°´, `h´)
			computeRecursive(&f_h[0], &y_k_h[0], x_h_1, num_images/2);
			computeRecursive(&f_h[num_images/2], &y_k_h[num_images/2], x_h_2, num_images/2);
			for(int i=0; i < num_images / 2;i++)
				optimizeF_helper((optimizeF_helper_struct_t) {.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_2});
			for(int i=num_images/2; i < num_images; i++)
				optimizeF_helper((optimizeF_helper_struct_t) {.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_1});
			@_free_stack°
		} else {
			float* x_h_1 = y_k_h[1];
			float* x_h_2 = y_k_h[0];
			for(int i=0; i < num_images / 2;i++)
				optimizeF_helper((optimizeF_helper_struct_t) {.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_2});
			for(int i=num_images/2; i < num_images; i++)
				optimizeF_helper((optimizeF_helper_struct_t) {.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_1});
		}
		optimizeX(f_h, x, y_k_h, num_images);
	}
int main(int argc, char** argv) {
	IplImage* input_image = cvLoadImage(argv[1], 0); //CV_LOAD_IMAGE = 0, according to opencv2/highgui/highgui_c.h
	/*float* f_h[@DEF_NUM_IMGS°];
	float* y_k_h[@DEF_NUM_IMGS°];
	float* x;
	@dnl° TODO: implement the allocation of y_k_h...
	@dnl° TODO: eventually switch to texture-based reading of the integer-based input images to conserve memory and enable bigger optimizeX inputs (num_images) to speed it up (also check if the host-memory access speed/PCIe transfer speed is the bottleneck for large optimizeX inputs (num_images)
	computeRecursive(f_h, y_k_h, x, @DEF_NUM_IMGS°);*/
	return 0;
}
