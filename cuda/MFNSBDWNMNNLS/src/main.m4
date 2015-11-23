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
#include <unistd.h>
#include <sched.h>
@define(`stop', `@dnl°')
@changequote(`[', `]') stop ´´
@changequote([`], [´]
@dnl° define(`@CALL_APPEND´,`define(`@DEF_MYLIST°´,ifdef(`@DEF_MYLIST°´,`[changequote([,])@DEF_MYLIST°[,$1]changequote(`,´)]´,[$1]))´)
@define(`@DEF_concatVarSize°´, eval(`1 ** 8´))
@define(`@DEF_m°´, eval(`10 * 2´))@dnl° The non-monotonic NNLS solvers inner iteration count M, which has to be even.
@define(`@DEF_N_target_optimization_F°´, `1e-5´)@dnl° The target value to get |\nabla f| to before stopping the optimizations of F
@define(`@DEF_N_target_optimization_X°´, `1e-40´)@dnl° The target value to get |\nabla f| to before stopping the optimizations of X
@define(`@DEF_BETA_F°´, `0.5´)@dnl° The non-monotonic NNLS solvers tweaking parameter β
@define(`@DEF_SIGMA_F°´, `0.5´)@dnl° The non-monotonic NNLS solvers tweaking parameter σ
@define(`@CALL_GEWICHTUNG´. `((1-abs(($1)- (0.5f * (@DEF_storedSizeX° - 1))+0.5f)*(2.f/(@DEF_storedSizeX°+1)))* (1-abs(($2)- (0.5f * (@DEF_storedSizeX° - 1))+0.5f)*(2.f/(@DEF_storedSizeX°+1))))´)@dnl° The trusty macro to calculate the correct wheigh for a given coordinate
@define(`@DEF_BLOCk_TOO_HIGH_THREADS_XY°´, `if (blockIdx.x == gridDim.x -1 && threadIdx.x >= (((@DEF_PATCH_NUM_X° + 1) * @DEF_storedSizeX° / 2 @ifelse(`Y´, `$1´, `2 * @DEF_SIZE_HALF_F°´)) * ((@DEF_PATCH_NUM_Y° + 1) * @DEF_storedSizeX° / 2 @ifelse(`Y´, `$1´, `2 * @DEF_SIZE_HALF_F°´)) - (gridDim.x -1) * blockDim.x) -1)´) @dnl° This is only the opening if(), not the {} nor an else
@define(`@DEF_NUM_PATCHES°´, @eval(`@DEF_NUM_PATCHES_X° * @DEF_NUM_PATCHES_Y°´) @dnl° just the total count of patches
@define(`@DEF_F_SQRD°´, @eval(`@DEF_SIZE_F° ** 2´)) @dnl° the number of values in one f
@define(`@DEF_NUM_F_VALS°´, @eval(`@DEF_NUM_PATCHES° * @DEF_F_SQRD°´)) @dnl° the total number of values of all f
@define(`@DEF_SIZE_Y°´, @eval(`(@DEF_storedSizeX° / 2) ** 2 * (@DEF_NUM_PATCHES_X° + 1) * (@DEF_NUM_PATCHES_Y° + 1)´)) @dnl° the number of elements in an y-space

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
	@DEF_BLOCK_TOO_HIGH_THREADS_XY(`Y´)
		diff = 0;
	else {
		diff = y[index] - y_k[index];
		v_3[index] = diff;
	}
	@define(`@CALL_BUTTERFLY_REDUCTION´,`{ for( int i = 16; i >= 1; i /= 2) $1 += __shfl_xor($1, i 32);}´) @dnl° Basic additive XOR butterfly reduction across current warp on the given argument
	@define(`@CALL_BUTTERLFLY_BLOCK_REDUTCTION´, `{ //Reduction
		@CALL_BUTTERFLY_REDCTION($1)
		if(threadIdx.x%32==0) @ifelse(`s´, `$3´, `((float*) sharedPointer)[threadIdx.x / 32]´, `part_Sums[threadIdx.x / 32´) = $1;
		__syncthreads();
		if(threadIdx.x/32 == 0) {
			$1 = @ifelse(`s´, `$3´, `(threaIdx.x/32 > blockDim.x/32)? 0 : ((float*) sharedPointer)[threadIdx.x & 0x1f];´, `part_Sums[threadIdx.x & 0x1f];´)
			@CALL_BUTTERFLY_REUCTION($1)
			if (threadIdx.x%32) {$2}}´) @dnl° whole 1024 Threads (in x index only) reduction across the block on $1, eecuting $2 at the end in the first thread of the block.
	@CALL_BUTTERFLY_BLOCK_REDUCTION(`diff´, `f_n_part_sums[blockIdx.x] = diff;
		__threadfence();
		unsigned int value = atomicInc(count, gridDim.x);
		isLastBlockDone = (value == (gridDim.x - 1));´) @dnl° use that reduction!
	__syncthreads();
	if (isLastBlockDone) {
		if (gridDim.x >  blockDim.x) {
			value = 0;

			for (int x=0; (gridDim.x % blokDim.x) == 0 ? x < (gridDim.x / blockDim.x) : x <= (gridDim.x / blockDim.x); x++)
				value += (gridDim.x % blockDim.x) == 0 || threadIdx.x * blockDim.x < gridDim.x ? f_n_part_sums[threadIdx.x * blockDim.x] : 0;
		}
		@CALL_BUTTERFLY_BLOCK_REDUCION(value, `*f_n = value;
		*count = 0;´) @dnl° reduction across the partial sums
	}
}

__global__ void kernel_nabla_tilde_Gets_nabla_capped_with_rule( float* __restrict__ vec_nabla_tide_f, float* __restrict__ vec_nabla_f, float* __restrict__ vec_X) {
	@DEF_BLOCK_TOO_HIGH_THEADS_XY(`X´)
		return;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	vec_nabla_tilde_f[i] = (0 < vec_nabla_f[i] && 0 == vec_x[i]) ? 0 : vec_nabla_f[i];
}

__device__ @DEF_FFT_PRECISION(`R´) load_f_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ shared_Ptr) {
	@define(`@CALL_SPLIT_concatVar´, `int xPos = (offset / @DEF_concatVarSize°) & @DEF_SIZE_concatVarSize°;
	int yPos = offset & @DEF_concatVarSize°;
	int patchNum = offset / @DEF_cpncatVarSize° / @DEF_concatVarSize°;´)
	@define(`@DEF_xPatchOffset°´, @eval(((@DEF_storedSizeX° ** 2) * (@DEF_NUM_PATCHES_Y° + 1)) / 2))
	@define(`@DEF_yPatchOffset°´, @eval(@DEF_storedSizeX° / 2))
	@define(`@CALL_RESTRICT_WITH_PADDING´, `@CALL_SPLIT_concatVar
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
	}´, `X´, `$1´, `int zero_space[4]; @dnl° 0 <= x < 1 && 2 <= y < 3
	int xPatch = patchNum / @DEF_NUM_PATCHES_Y°;
	int yPatch = patchNum - xPatch * @DEF_NUM_PATCHES_Y°;
	if(0 == xPatch) { @dnl° x = 0 border
		zero_space[0] = @eval(@DEF_SIZE_HALF_F° + (@DEF_storedSizeX° / 2));
		zero_space[1] = @ifelse(`Y´, `$2´, `@DEF_FFT_SIZE°´, `X´, `$2´, `@eval(@DEF_SIZE_HALF_F° + @DEF_storedSizeX°)´);
	} else if (@DEF_NUM_PATCHES_Y° - 1 == xPatch) { @dnl° x = MAX border
		zero_space[0] = @ifelse(`Y´, `$2´, `0´, `X´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[1] = @eval(@DEF_SIZE_HALF_F° + (@DEF_storedSizeX° / 2));
	} else { @dnl°  no x border
		zero_space[0] = @ifelse(`Y´, `$2´, `0´, `X´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[1] = @ifelse(`Y´, `$2´, `@DEF_FFT_SIZE°´, `X´, `$2´, `@eval(@DEF_SIZE_HALF_F° + @DEF_storedSizeX°)´);
	}

	if (0 == yPatch) { @dnl° y = 0 border
		zero_space[2] = @eval(@DEF_SIZE_HALF_F° + (@DEF_storedSizeX° /2));
		zero_space[3] = @ifelse(`Y´, `$2´, `@DEF_FFT_SIZE°´, `X´, `$2´, `@eval(@DEF_SIZE_HALF_F° + @DEF_storedSizeX°)´);
	} else if (@DEF_NUM_PATCHES_X° - 1 == yPatch) { @dnl° y = MAX border
		zero_space[2] = @ifelse(`Y´, `$2´, `0´, `X´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[3] = @eval(@DEF_SIZE_HALF_F° + (@DEF_storedSizeX° /2));
	} else { @dnl° no y border
		zero_space[2] = @ifelse(`Y´, `$2´, `0´, `X´, `$2´, `@DEF_SIZE_HALF_F°´);
		zero_space[3] = @ifelse(`Y´, `$2´, `@DEF_FFT_SIZE°´, `X´, `$2´, `@eval(@DEF_SIZE_HALF_F° + @DEF_storedSizeX°)´);
	}

	int patcOffset = @DEF_xPatchOffset° * xPatch + @DEF_yPatchOffset° * yPatch;
	if (zero_space[0] <= xPos && xPos < zeroSpace[1] && zero_space[2] <= yPos && yPos < zero_space[3]) {
		int yPosStored = yPos - @DEF_SIZE_HALF_F°;
		int xPosStored = xPos - @DEF_SIZE_HALF_F°;
		$3
	}
	@dnl° TODO: missing y and rest of it.
	@dnl° TODO: complete for the zeroSpace[4] way in X and use $2, $3, etc. to select the right way, but only as many arguments as needed.
	´)´)
	@CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
	return ((@DEF_FFT_PRECISION(`R´)*) dataIn)[(@DEF_SIZE_F° * @DEF_SIZE_F°) *  patchNum + @DEF_SIZE_F° * xPosStored + yPosStored] * @CALL_GEWICHTUNG(`xPos - @DEF_SIZE_HALF_F° - (@DEF_storedSizeX°/2)´, `yPos - @DEF_SIZE_HALF_F° - (@DEF_storedSizeX°/2)´);
}

__device__ void store_f_T_p_conj_fft_X(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = @ifelse(`float´, @DEF_FFT_PRECISION_TYPE°, `cuConjf´, `double´, @DEF_FFT_PRECISION_TYPE°, `cuConj´)(elment); @dnl° TODO: insert right cmmand/
}

__device__ @DEF_FFT_PRECISION(`C´) load_F_X_m_F_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __retrict__ sharedPointer) {
	return @ifelse(`float´, @DEF_FFT_PRECISION_TYPE°, `cuCmulf´)(((@DEF_FFT_PREC° ISION(`C´)*) (dataIn))[offset], ((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);  @dnl° TODO: include double precision as an option for the commplex multipliction.
}

__device__ void store_v_4_F_T_v_4_p_weight_half_v_1_X(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	@CALL_RESTRICT_WITH_PADDING(`X´, `Y´, `element *= .5f * @CALL_GEWICHTUNG(`xPosStored - (@DEF_storedSizeX° /2)´, `yPosStored - (@DEF_storedSizeX° /2)´);
	atomicAdd(&((@DEF_FFT_PREISION(`R´)*) (dataOut))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored], element);
´)
}

__device__ @DEF_FFT_PRECISION(`R´) load_x_p_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrit__ sharedPtr) {
	@DEF_FFT_PRECISION(`R´) ret;
	@CALL_RESTRICT_WITH_PADDING(`X´, `Y´, `ret = ((@DEF_FFT_PRECISION(`R´)*) dataIn)[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored];
		ret *= @CALL_GEWICHTUNG(`xPosStored´, `yPosStored´);
´) else
		ret = 0;
	return ret;
}

__device__ @DEF_FFT_PRECISION(`R´) load_f_X_1_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	@CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
	return ((@DEF_FFT_PRECISION(`R´)*) dataIn)[(@DEF_SIZE_F° * @DEF_SIZE_F°) * patchNum + SIZE_F * xPosStoed + yPosStored];
}

__device__ @DEF_FFT_PRECISION(`R´) load_v_3_X_T_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	@CALL_RESTRICT_WITH_PADDING(`X´, `X´, `return ((@DEF_FFT_PRECISION(`R´)*) (daaIn))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored];´) else
		return 0;
}

__device__ void store_f_X_y_p_v_1_F(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	@CALL_RESTRICT_WITH_PADDING(`X´, `X´, `atomicAdd(&((@DEF_FFT_PRECISION(`R´)*) (dataOut))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored], element * ((float) (1. / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE°))))´)
}

__device__ void store_f_X_fft_m_x_F(void+ __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __retrict__ sharedPointer) {
	((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = cuCmulf(((@DEF_FFT_PRECISION(`C´)*) (datOut))[offset], ((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);
}

__device__ void store_f_X_T_fft_m_x_F(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restict__ callerInfo, void* __retrict__ sharedPointer) {
	((@DEF_FFT_PECISION(`C´)*) (dataOut))[offset] = cuCmulf(((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset], cuConjf(((DEF_FFT_PREISION(`C´)*) (callerInfo))[offset]));
}
@define(`@DEF_STORE_REDUCE_CALL°´, `@ifelse(`11´, `$1´, `store_f_X_T_1_nabla_tilde_f_uneven_b_F´, `12´, `$1´, `store_f_X_T_1_nabla_tilde_f_even_b_F´, `21´, `$1´, `store_f_X_T_2_delta_tilde_f_even_b_F´, `22´, `$1´, `store_f_X_T_2_delta_tilde_f_uneven_b_F´)´)
@define(`@DEF_STORE_REDUCE_DEF°´, `__device__ void @DEF_STORE_REDUCE_CALL(`$1$2´) (void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
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
	@ifelse(`1´, $1´, `if (element > 0 &&  0 == vec_f_o[index])
		nable_tilde_f = 0;
	else
		nabla_tilde_f = ((float) .5 / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE°)) * element,

	f = vec_f_o[index] - inform_struct->alpha * inform_struct->beta * nabla_tilde_f;
	value = vec_nabla_tilde_f_o[index] * (vec_f_o[index] - vec_f[index]);´, `value = inform_struct->vec_nabla_f_o[index] * element * ((float) 1. / (@DEF_FFT_SIZE° * _DER_FFT_SIZE));´)

	sumItUp:

	@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, `@ifelse(`1´, $1´, `inform_struct->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d´, `ìnform_struct->nabla_f_scalar_prod_delta_f_part_sums´)[gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockDim.y + blockIdx.x] = value;
	inform_struct->block_num = gridDim.x * gridDim.y * gridDim.z;
		inform_struct->block_size = blockDim.x * blockDim.y * blockDim.z;´, `s´)

	value = @ifelse(`1´, `$1´, `nabla_tilde_f * nabla_tilde_f;´, `element * element * ((float) 1. / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE° * @DEF_FFT_SDIZE° * @DEF_FFT_SIZE°));´)
	@ifelse(`2´, `$2´, `@CALL_BUTTERFLY_BLOCK_REDUCTION(`value´, 	`@ifelse(`1´, `$1´, `inform_struct->abs_vec_nabla_f_part_sums´, `abs_vec_delta_f_part_sums´)[gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x] = value;´, `s´)´)

	if (isF)
		vec_f_o[index] = f;
}´)

@DEF_STORE_REDUCE_DEF°(1, 2)

@DEF_STORE_REDUCE_DEF°(1, 1)

@DEF_STORE_REDUCE_DEF°(2, 1)

@DEF_STORE_REDUCE_DEF°(2, 2)

@dnl° that have been all the function definitions for the device side, except the not refactored, but to be coded, device side reduction/summation code for those reductions previously done in host code (to seriously reduce host<->device traffic
@define(`@COMPOUND°´, @ifelse(`L´, $1, `load´, `S´, `$1´, `store´)`_$3´)
@define(`@CALL_ALLOC_CB´, `__device__ cufftCallback´@ifelse(`L´, `$1´, `Load´, `S´, `$1´, `Store´)`$2 _d_´@COMPOUND°($@)`cufftCallback´@ifelse(`L´, `$1´, `Load´, `S´, `$1´, `Store´)`$2 _h_´@COMPOUND°($@)`_$3;@divert(1)cudaMemcpyFromSymbol(&_h_´@COMPOUND°($@)`, _d_´@COMPOUND°($@)`, sizeof(_h_´@COMPOUND°($@)`));
@divert(0)´) stop ´)´)

@dnl° TODO: insert all the @CALL_ALLOC_CB

void getCallbacks() {
	@undivert(1)}
@define(`@CALL_ROUND_BLOCK_SIZE_UP´, `((($1) % ($2) ? ($1) / ($2) : ($1) / ($2) + 1))´)
int setFloatDeviceZero(float* data, size_t count, int blocksize, cudaStream_t stream) {
	kernel_set_float_zero<<<@CALL_ROUND_BLOCK_SIZE_UP(`count´, `blocksize´), blocksize, 0, stream>>>(data, count % blocksize);
	return 0;
}

void optimizeFcallback(cudaStream_t stream,  cudaError_t status, void* __restrict__ userData) {
	struct streamCallback_information *informations = ((struct streamCallback_informations*) userData);
	if (informations->b % 2 == 0) {
		float abs_nabla_f = 0;
		float delta_nabla_f = 0;
		if (*(informations->f_n_h) < @DEF_N_SOLL_F°) {
			@dnl° optimization is finished
			informations->finished = true;
			return;
		}
		for (int i=0; i < informations->helper_struct_b->block_num; i++) {
			abs_nabla_f += informations->part_sums_var_h[i];
			delta_nabla_f += informations->delta_nabla_f_part_sums_h[i];
		}
		informations->helper_struct_h->alpha = abs_nabla_f / delta_nabla_f;
	} else {
		float abs_delta_f = 0;
		float delta_nabla_f = 0;
		for (int i = 0; i < informations->helper_struct_h->block_um; i++) {
			abs_delta_f += informations->part_sums_var_h[i];
			delta_nabla_f += informations->delta_nabla_f_part_sums_h[i];
		}
		informations->helper_struct_h->alpha = delta_nabla_f / abs_delta_f;
	}
	float complicatedSums = 0;
	for (int i = 0; i < informations->heler_struct_h->block_num; i++) {
		complicatedSums += (informations->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h)[i];
	if (informations->f_o_h - informations->f_n_h <= @DEF_SIGMA_F° * complicatetSums)
		informations->helper_struct_h->beta *= @DEF_BETA_F°;
	informations->f_o_h = informations->f_n_h;
}

int optimizeF(float* f_h, float* x_h, float* y_k_h, cudaStream_t stream) {
	int dev;
	@define(`@_free_stack°´, `@ifdef(`@_free_stack1°´, `@_free_stack1°@popdef(`@_free_stack_1°´)@_free_stack°´)´)
	@define(`@DEF_CU_MALLOC°´, `$2* $1 = NULL;
	cudaMalloc´@ifelse(`h´, `$4´, `Host@pushdef(`@_free_stack1°´, `cudaFreeHost($1);
´)´, `@pushdef(`@_free_stack1°´, `$1´)´)`((void**) &$1, sizeof($2) * $3);´) @dnl° $1 = [device] pointer name, $2 = [device] pointer type (without the '*'), $3 = number of elements to allocate[, $4 = h (to allocate host space)
	@define(`@DEF_CU_MALLOC_HTDC°´, `@DEF_CU_MALLOC($@)@ifelse(`´, `$7´,,`
$7´)@divert(1)
	cudaMemcpyAsync($1, $5, sizeof($2) * $3, cudaMemcpyHostToDevice, $6);@divert(0)@ifelse(`´, `$8´,,`@divert(2)
$8@divert(0)´)´) @dnl° $1 = device pointer name, $2 = device pointer type (without the '*'), $3 = number of elements to allocate, $4 = '' (just jump with a double ','), $5 = host pointer name, $6 = stream, $7 = optional (somthing to execute after the allocation and before scheduling the copy for the bunch of copys, $8 = optional (to execute after copying)
	@DEF_CU_MALLOC_HTDC(`f_d´, `float´, `@DEF_NUM_F_VALS°´,, `f_h´, `stream´)
	@DEF_CU_MALLOC_HTDC(`y_k_d´, `float´, `@DEF_SIZE_Y°´,, `y_k_h´, `stream´)
	@DEF_CU_MALLOC_HTDC(`helper_struct_d´, `store_f_X_T_1_informations´, 1,, `helper_struct_h´, `stream´, `@DEF_CU_MALLOC(`helper_struct_h´, `store_f_X_T_1_informations´, 1, `h´)
	helper_struct_h->alpha = 0.5;
	helper_struct_h->beta = 0.5;´)
	@DEF_CU_MALLOC(`helper_struct_h->vec_f_o´, `float´, `@DEF_NUM_F_VALS°´)
	@DEF_CU_MALLOC(`helper_struct_h->vec_nabla_f_o´, `float´, `@DEF_NUM_F_VALS°´)
	@DEF_CU_MALLOC(`x_p_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(`@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°´))
	@DEF_CU_MALLOC_HTDC(`v_3_d´, `float´, `@DEF_SIZE_Y°´,,`x_h´, `stream´,,`cufftExecR2C(plan_x_p_F, v_3_d, x_p_d);
	setFloatDeviceZero(v_3_d, `@DEF_SIZE_Y°´, 128, stream);´)
	@DEF_CU_MALLOC(`y_d´, `float´, `@DEF_SIZE_Y°´)
	@DEF_CU_MALLOC(`v_tmp_cmplx_d´, `@DEF_FFT_PRECISION(`C´)´, @eval(`@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1) * @DEF_NUM_PATCHES°´))
	@DEF_CU_MALLOC(`f_n_d´, `float´, 1)
	@DEF_CU_MALLOC(`count_d´, `unsigned int´, 1)
	@DEF_CU_MALLOC(`f_n_part_sums_d´, `float´, @CALL_ROUND_BLOCK_SIZE_UP(@DEF_SYZE_Y°, 1024))
	@DEF_CU_MALLOC(`(helper_struct_h->abs_vec_nabla_f_part_sums)´, `float´, @DEF_NUM_F_VALS°)
	@DEF_CU_MALLOC(`(helper_struct_h->abs_vec_delta_f_part_sums)´, `float´, @DEF_NUM_F_VALS°)
	@DEF_NUM_MALLOC(`(helper_Struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d)´, `float´, @DEF_NUM_F_VALS°)
	@DEF_CU_MALLOC(`(helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums)´, `float´, 32768)' @dnl° TODO: care and decide about helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums size '32768'
	streamCallback->finished = false;
	streamCallback->helper_struct_d = helper_Struct_D;
	streamCallback->helper_Struct_h = helper_Struct_h;
	streamCallback->f_n_h = f_n_h;
	@dnl° TODO: check for memset to f_n_d (if it is necessary)
	@define(`@nargs´, `$#´) @dnl° just emit the number of arguments given. Usefull to determine the size of a grouped argument.
	@define(`@_echo_q´, `$@´) @dnl° just a macro to ecpand into all the args, qouted. Usefull to expand a grouped argument.
	@define(`@_CB_PLAN_STMT´, `@_CB_PLAN_STMT1(`$4´, `$2´, `$3´, __echo_q$1)´)
	@define(`@_CB_PLAN_STMT1´, `cufftXtSetCallback(plan_$2, ((void**) &_h_@ifelse(`´, `$4´, `@ifelse(`l´, `$1´, `load_´, `s´, `$1´, `store_´)$2´, $4)), CUFFT_CB_@ifelse(`lC´, `$1$3´, `LD_COMPLEX´, `lR´, `$1$3´, `LD_REAL´, `sC´, `$1$3´, `ST_REAL´, `sR´, `$1$3´, `ST_COMPLEX´), @ifelse(`´, `$5´, `NULL´, `((void**) &$5_d)´));@ifelse(`5´, `$#´, `´, `
		cufftXtSetCallbackSharedSize(plan_$2, CUFFT_CB_@ifelse(`lC´, `$1$3´, `LD_COMPLEX´, `lR´, `$1$3´, `LD_REAL´, `sC´, `$1$3´, `ST_REAL´, `sR´, `$1$3´, `ST_COMPLEX´), $6);´)´)
		@dnl° TODO: insert the shared memory reservation call (with semicolon), as well as the following at the end: ´)´) m4_dnl <insert documentation here>
	@define(`@DEF_CUFFT_HANDLE°´, `cufftHandle plan_$1;
	{
		cufftCreate(&$1);
		int inembed[] = { 1, @ifelse(`C´, `$2´, @eval(`@DEF_FFT_SIZE° / 2 + 1´), @DEF_concatVarSize°) };
		int onembed[] = { 1, @ifelse(`C´, `$2´, @DEF_concatVarSize°, @eval(`@DEF_FFT_SIZUE° / 2 + 1´)) };
		int n[] = { @DEF_FFT_SIZE°, @DEF_FFT_SIZE° };
		cufftPlanMany(&$1, 2, n, inembed, 1, @ifelse(`C´, `$2´, `@eval(`@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1)´)´, `R´, `$2´, `@eval(`@DEF_concatVarSize° ** 2´)´), onembed, 1, @ifelse(`C´, `$2´, `@eval(`@DEF_concatVarSize° ** 2´), CUFFT_C2R´, `R´, `$2´, `@eval(`@DEF_FFT_SIZE° * (@DEF_FFT_SIZE° / 2 + 1)´), CUFFT_R2C´), @DEF_NUM_PATCHES°);
		cufftSetStream($1, $3);
		@ifelse(_nargs$4, `1´, `´, `@_CB_PLAN_STMT($4, `$1´, `$2´, `l´)´)
		@ifelse(_nargs$5, `1´, `´, `@_CB_PLAN_STMT($5, `$1´, `$2´, `s´)´)
	}´)
	@dnl° $1 = name of the plan, without the leading plan_, $2 = 'C' if C2R; 'R' if R2C, $3 = name of the stream to execute in, $4 = ([[loadCallbackName <without the leading _h_, if omitted: _h_load_$1>], [callerInfo device pointer<without the trailing _d, if omitted: NULL>] <to omit: leave the parenthesis empty and omit the comma in between>]), \
	@dnl° $5 = ([[storeCallbackName <without the leading _h_, if omitted: _h_store_$1>], [callerInfo device pointer <without the trailing _d, if omited: NULL>][, size to request for shared memory allocation <inclusive any sizeof(...) factors>]<to omit: leave the parenthesis empty and omit the comma in between>])
	@undivert(1)
	@DEF_CUFFT_HANDLE°(`x_p_F´, `R´, `stream´, (,), ())
	@DEF_CUFFT_HANDLE°(`f_X_1_l_F´, `R´, `stream´, (,), (`f_X_fft_m_x_F´, `x_p´))
	@DEF_CUFFT_HANDLE°(`f_X_1_s_F´, `C´, `stream´, (), (`f_X_y_p_v_1_F´,))
	@DEF_CUFFT_HANDLE°(`f_X_T_l_F´, `R´, `stream´, (`v_3_X_T_F´,), (`f_X_T_fft_m_x_F´, `x_p´))
	@DEF_CUFFT_HANDLE°(`f_X_T_!_nabla_tilde_f_even_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@DEF_CUFFT_HANDLE°(`f_X_T_1_nabla_tilde_F_uneven_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@DEF_CUFFT_HANDLE°(`f__X_T_2_delta_tilde_f_even_b_F´, `C´, `stream´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@DEF_CUFFT_HANDLE°(`f_X_T_2_delta_tilde_f_uneven_b_F´, `C´, (), (, `helper_struct´, `sizeof(float) * 32´))
	@undivert(2)
	do {
		for (int b=0; b < @DEF_m°; b++) {
			cufftExecR2C(plan_f_X_1_l_F, f_d, v_tmp_cmplx_d);
			setFloatDeviceZero(y_d, @DEF_SIZE_Y°, 128, stream);
			cufftExecC2R(plan_f_X_1_s_F, v_tmp_cmplx_d, y_d);
			kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr<<<@CALL_ROUND_BLK_SIZE_UP(`@DEF_SIZE_Y°´, `1024´), 1024, 0, stream>>>(v_3_d, y_d, y_k_d, f_n_part_sums_d, f_n_d, count_d);
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
			if (b == 0) {
				while(cudaErrNotFinished == cudaStreamQuery(stream))
					usleep(@DEF_SLEEP_TIME_POLL°); @dnl° TODO: convert from current runtime-based allocation mechanism to precalculated one, thereby preventing the hangup happening here.
				@DEF_CU_MALLOC(`delta_nabla_f_part_sums_h´, `float´, `helper_struct_h->block_num´, `h´) @dnl° TODO: further research the reason of using an additional '0' as the last argument to this call
				@DEF_CU_MALLOC(`part_sums_var_h´, `float´, `helper_struct_h->block_num´, `h´)
				@DEF_CU_MALLOC(`(streamCallback->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h)´, `float´, `helper_struct_h->block_num´, `h´)
				streamCallback->delta_nabla_f_part_sums_h = delta_nabla_f_part_sums_h;
				streamCallback->part_sums_var_h = part_sums_var_h;
			}
			cudaMemcpyAsync((void*) delta_nabla_f_part_sums_h, (void*) helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			if (b % 2 == 0)
				cudaMemcpyAsync((void*) part_sums_var_h, (void*) helper_struct_h->abs_vec_nabla_f_part_sums, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			else
				cudaMemcpyAsync((void*) part_sums_var_h, (void*) helper_struct_h->abs_vec_delta_f_part_sums, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			cudaMemcpyAsync((void*) f_n_h, (void*) f_n_d, sizeof(float), cudaMemcpyDeviceToHost, stream);
			cudaMemcpyAsync((void*) streamCallback->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h, (void*) helper_struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d, sizeof(float) * helper_struct_h->block_num, cudaMemcpyDeviceToHost, stream);
			cudaStreamAddCallback(stream, (cudaStreamCallback_t) optimizeFcallback, (void*) streamCallback, 0);
			if (b % 2 == 0) {
				while(cudaErrNotFinished == cudaStreamQuery(stream))
					usleep(@DEF_SLEEP_TIME_POLL°);
				if (streamCallback->finished)
					goto end_loop;
			}
			cudaMemcpyAsync(helper_struct_d, helper_struct_h, sizeof(store_f_X_T_1_informations), cudaMemcpyHostToDevice, stream);
		}
	} while (true); @dnl° TODO: check if while (true) is really the right thing to do here.
	end_loop: cudaMemcpyAsync((void*) f_h, (void*) f_d, sizeof(float) * @DEF_NUM_F_VALS°, cudaMemcpyDeviceToHost, stream);
	while(cudaErrNotFinished == cudaStreamQuery(stream))
		usleep(@DEF_SLEEP_TIME_POLL°);
	@_free_stack°
	return 0;
}
@dnl° __device__ @DEF_FFT_PRECISION(`R´) load_f_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
@dnl°	@CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
@dnl°	return @CALL_GEWICHTUNG(`xPos´, `yPos´) * ((@DEF_FFT_PRECISION(`R´)*) dataIn)[(@DEF_SIZE_F° * @DEF_SIZE_F°) * patchNum + SIZE_F * xPosStoed + yPosStored];
@dnl°} @dnl° TODO: check why this is already done above. seems kinda strange, but it may be the first part of optimizeX() I did back then... also this is probably not the right call to @CALL_GEWICHTUNG(), as I did it differently above.


__device__ @DEF_FFT_PRECISION(`R´) load_x_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrit__ sharedPtr) {
	@DEF_FFT_PRECISION(`R´) ret;
	@CALL_RESTRICT_WITH_PADDING(`X´, `Y´, `ret = ((@DEF_FFT_PRECISION(`R´)*) dataIn)[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored];
´) else
		ret = 0;
	return ret;
}
__device__ @DEF_FFT_PRECISION(`C´) load_x_p_cmplx_mul_f_p(void+ __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __retrict__ sharedPointer) {
	((@DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = cuCmulf(((@DEF_FFT_PRECISION(`C´)*) (datOut))[offset], ((@DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);
} @dnl° convert the parameters and rest from store to load.

__device__ void store_y_plus_y_X(void* __restrict__ dataOut, size_t offset, @DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	@CALL_RESTRICT_WITH_PADDING(`X´, `Y´, `atomicAdd(&((@DEF_FFT_PRECISION(`R´)*) (dataOut))[patchOffset + xPosStored * @DEF_storedSizeX° + yPosStored], element * ((float) (1. / (@DEF_FFT_SIZE° * @DEF_FFT_SIZE°))))´)
}
int optimizeX(float** f_h, float* x_h, float** y_k_h, int num_images){ @dnl° TODO: convert the symbolic code to actual code
	@dnl° TODO: make sure to not overuse memory space and memory access in the implementation
	@dnl° TODO: maybe eventually make this use host memory where applicable 
	cudaEvent_t events[num_images];
	cudaStream_t streams[num_images];
	cudaEvent_t helperEvents[2];
	cudaEventCreateWithFlags(&helperEvents[0], cudaEventDisableTiming);
	cudaEventCreateWithFlags(&helperEvents[1], cudaEventDisableTiming);
	for (int k = 0; k < num_images; k++) { @dnl° Create some events and streams, so we can better parallize the images where applicable.
		cudaEventCreateWithFlags(&events[k], cudaEventDisableTiming);
		cudaStreamCreate(&streams[k]);
	}
	{ // precompute F_{k,i,j} and F^T_{k,i,j}
		for (int k = 0; k < num_images; k++) {

			// f_p_k{_i,j} := 2Dfft(gewichtung(zuSchnipselGröße(f_k{_i,j})))
			// => F_{k,k,j}
			load_f_p_X() -> NULL [stream k]

			// f_t_p_k{_i,j} := conj(2Dfft(zuSchnipselGröße(f_k{_i,j})))
			// => F^T_{k,i,j}
			load_f_X_1_F() -> store_f_T_p_conj_fft_X() [stream k]

			cudaEventRecord(events[k], streams[k]);
		}
	}
	for (int k = 0; k < num_images; k++)
		cudaStreamWaitEvent(streams[0], events[k], 0);

	do {
		for (int b = 0; b < @DEF_m°; b++) {
			// v_4 = 0
			// X''{_i,j} = 2Dfft(X{_i,j})
			load_x_p_X() -> NULL [stream 0]

			cudaEventRecord(helperEvents[0], streams[0]);
			for (int k = 0; k < num_images; k++) { @dnl° TODO: use streams and/or pthreads for using the parallelism avaiable here.

				cudaStreamWaitEvent(streams[k], helperEvents[0], 0);

				// y_k{i,j} = y_k{_i,j} + 2Difft(X''{_i,j} * F_k{_i,j})
				// => F_k
				load_F_X_m_F_X() -> store_y_plus_y_X() [stream k]

				// v_3 = clip^X_y(y_k) @dnl° Already done via the restriction to `X´, `Y´ in  the last statement.
				// v_3 = v_3 - y'_k
				kernel_v_3_gets_y_min...() @dnl° TODO: make sure this gets a new f_n{_i} and count{_i} for each input image
				// v_4{_i,j} = v_4{_i,j} + .5 * gewichtung(2Difft(2Dfft(v_3{_i,j}) * f_t_p_k{_i,j}))
				// => F^T_k
				load_v_3_X_T_F() -> NULL; load_F_X_m_F_X() -> store_x_plus_x_weights_X() [stream k] @dnl° *= .5; see: nabla_f_to_nabla_tilde_f_kernel_X

				cudaEventRecord(events[k], streams[k]);
			}

			for (int k = 0; k < num_images; k++)
				cudaStreamWaitEvent(streams[0], events[k], 0);

			// nabla_f = v_4

			// nabla_tilde_f = nabla_F * { nabla_F > 0 && x == 0}
			// => active set classification

			// v_4 = 0

			// nabla_tilde_f'{_i,j} = 2Dfft(nabla_tilde_f{_i,j})

			{ @dnl° TODO: this will be a single kernel. see: nabla_f_to_nabla_tilde_f_kernle_X
				v_4 *= .5; @dnl° grabbed this to not do it in store_y_plus_y_X()
				x_o = X;
				nabla_tilde_f = v_4 > 0 && 0 == X ? 0 : v_4;
				VX = x_o - \beta * \alpha * nabla_tilde_f;
				scalar_prod__bo_nabla_f__bo_x_o_min_X__bc__bc = scalar_prod(nabla_f_o, (x_o - X));
				nabla_f_o = nabla_tilde_f;
			} [stream 0]

			cudaEventRecord(helperEvents[0], streams[0]);

			cudaStreamWaitEvent(streams[1], helperEvents[0], 0);
			v_4 = 0; [stream 1]
			cudaEventRecord(helperEvents[1], streams[1]);

			nabla_tilde_f' = fft(nabla_tilde_f); [stream 0]
			cudaEventRecord(helperEvents[0], streams[0]);

			for (int k = 0; k < num_images; k++) {

				cudaStreamWaitEvent(streams[k], helperEvents[0], 0);
				// v_3{_i,j} = v_3{_i,j} + 2Difft(nabla_tilde_f'{_i,j} * f_p{_i,j})
				// => F_k
				load_F_X_m_F_X() -> store_y_plus_y_X() @dnl° TODO: check if we need a different plan for this or if we can use the same one.

				// v_3 = clip^X_y(v_3) @dnl° Already done via the restriction to `X´, `Y´ in  the last statement.

				cudaStreamWaitEvent(streams[k], helperEvents[1], 0);
				// v_4{_i,j} = v_4{_i,j} + gewichtung(2Difft(2Dfft(v_3{_i,j}) * f_t_p_k{_i,j}))
				// => F^T_k
				load_v_3_X_T_F() -> NULL; load_F_X_m_F_X() -> store_x_plus_x_weights_X()

				cudaEventRecord(events[k], streams[k]);

			}
			// delta_f_tilde{_i,j} = v_4

			// X = X - \beta * \alpha * nalba_f_tilde @dnl° DONE
			// => Update

			for (int k = 0; k < num_images; k++)
				cudaStreamWaitEvent(streams[0], events[k], 0);

			{ @dnl° TODO: incorporate everything from here on into the new kernel "delta_nabla_f_tilde", so we can make use of the reduction for {scalar_prod(nabla_tilde_f, delta_tilde_f); Sum[f_n{_i}, {i, 0, num_images-1}], |nabla_tilde_f|^2, |delta_tilde_f|^2} and instantly use the result. In case of n_a <= n_s set a pointer (in mapped host memory?) and later use that to decide wether to continue optimizeing or not.
				// delta_nabla_f_tilde = scalar_prod(nabla_f_tilde, delta_f_tilde)
				delta_nabla_f_tilde = scalar_prod(nabla_tilde_f, delta_tilde_f);

				if (b % 2 == 0) {
					// n_a = |nabla_f_tilde|^2
					if (n_a <= n_s)
						// return X;
					else
						// a = n_a / delta_nabla_f_tilde
				} else
					// a = delta_nabla_f_tilde / |delta_f_tilde|^2

				// f_n = sum^{num_images}_{k=0}(|y_k - y_k'|^2)
				if (f_o - f_n <= \sigma * scalar_prod(nabla_f_o, (x_o -X)) ) { @dnl° scalar_prod(nabla_f_o, (x_o -X)) see: nabla_f_to_nabla_tilde_f_kernel_X
					// \beta = \eta * \beta
				}

				// f_o = f_n @dnl° see: delta_nabla_f_tilde_kernel_X
				// nabla_f_o = nabla_tilde_f @dnl° see: delta_nabla_f_tilde_kernel_X
			} [stream 0]
			// x_o = X @dnl° see: nabla_f_to_nabla_tilde_f_kernel_X
		}
	}
	cudaEventDestroy(&helperEvents[0]);
	cudaEventDestroy(&helperEvents[1]);
	for (int k = 0; k < num_images; k++) { @dnl° Create some events and streams, so we can better parallize the images where applicable.
		cudaEventDestroy(&events[k]);
		cudaStreamCreate(&streams[k]);
	}
}
typedef struct optimizeF_helper_struct {
	float* f_h;
	float* y_k_h;
	float* x_h
void optimizeF_helper(void* datav) {
	optimizeF_helper_struct_t data = (optimizeF_helper_struct_t*) datav;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	optimizeF(data->f_h, data->y_k_h, data->x_h, stream);
	while(cudaErrorNotReady == cudaStreamQuery(stream)) sched_yield();
	cudaStreamDestroy(stream);
	return;
}
int computeRecursive(float** f_h, float** y_k_h, float* x, int num_images){
		pthread_t* threads;
		threads = (pthread_t*) malloc(sizeof(pthread_t) * num_images);
		if (num_images > 2) {
			@DEF_CU_MALLOC(`x_h_1´, `float´, `@DEF_SIZE_Y°´, `h´)
			@DEF_CU_MALLOC(`x_h_2´, `float´, `@DEF_SIZE_Y°´, `h´)
			optimizeRecursive(f_h[0], y_k_h[0], x_h_1, num_images/2);
			optimizeRecursive(f_h[num_images/2], y_k_h[num_images/2], x_h_2, num_images/2);
			for(int i=0; i < num_images / 2;i++)
				pthread_create(&(threads[i]), NULL, (void*) &optimizeF_helper, (void*) &(optimizeF_helper_struct_t){.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_2});
			for(int i=num_images/2; i < num_images; i++)
				pthread(_create(&(threads[i]), NULL, (void*) &optimizeF_helper, (void*) &(optimizeF_helper_struct_t){.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_1});
			for(int i=0; i < num_images; i++)
				pthread_join(threads[i], NULL);
			@_free_stack°
		} else {
			x_h_1 = y_k_h[1];
			x_h_2 = y_k_h[0];
			for(int i=0; i < num_images / 2;i++)
				pthread_create(&(threads[i]), NULL, (void*) &optimizeF_helper, (void*) &(optimizeF_helper_struct_t){.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_2});
			for(int i=num_images/2; i < num_images; i++)
				pthread(_create(&(threads[i]), NULL, (void*) &optimizeF_helper, (void*) &(optimizeF_helper_struct_t){.f_h = f_h[i], .y_k_h = y_k_h[i], .x_h = x_h_1});
			for(int i=0; i < num_images; i++)
				pthread_join(threads[i], NULL);
		}
		optimizeX(f_h, x, y_k_h, num_images);
		free(threads);
	}
int main(void) {
	float* f_h[@DEF_NUM_IMGS°];
	float* y_k_h[@DEF_NUM_IMGS°];
	float* x;
	@dnl° TODO: implement the allocation of y_k_h...
	@dnl° TODO: eventually switch to texture-based reading of the integer-based input images to conserve memory and enable bigger optimizeX inputs (num_images) to speed it up (also check if the host-memory access speed/PCIe transfer speed is the bottleneck for large optimizeX inputs (num_images)
	computeRecursive(f_h, y_k_h, x, @DEF_NUM_IMGS°);
