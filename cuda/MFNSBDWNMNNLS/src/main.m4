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
m4_define(`_DEF_BLOCk_TOO_HIGH_THREADS_XY´, `if (blockIdx.x == gridDim.x -1 && threadIdx.x >= (((_DEF_PATCH_NUM_X + 1) * _DEF_storedSizeX / 2 m4_ifelse(`Y´, `$1´, `2 * _DEF_SIZE_HALF_F´)) * ((_DEF_PATCH_NUM_Y + 1) * _DEF_storedSizeX / 2 m4_ifelse(`Y´, `$1´, `2 * _DEF_SIZE_HALF_F´)) - (gridDim.x -1) * blockDim.x) -1)´) m4_dnl This is only the opening if(), not the {} nor an else
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
	m4_define(`_CALL_BUTTERFLY_REDUCTION´,`{ for( int i = 16; i >= 1; i /= 2) $1 += __shfl_xor($1, i 32);}´) m4_dnl Basic additive XOR butterfly reduction across current warp on the given argument
	m4_define(`_CALL_BUTTERLFLY_BLOCK_REDUTCTION´, `{ //Reduction
		_CALL_BUTTERFLY_REDCTION($1)
		if(threadIdx.x%32==0) part_Sums[threadIdx.x / 32] = $1;
		__syncthreads();
		if(threadIdx.x/32 == 0) {
			$1 = part_Sums[threadIdx.x & 0x1f];
			_CALL_BUTTERFLY_REUCTION($1)
			if (threadIdx.x%32) {$2}}´) m4_dnl whole 1024 Threads (in x index only) reduction across the block on $1, eecuting $2 at the end in the first thread of the block.
	_CALL_BUTTERFLY_BLOCK_REDUCTION(`diff´, `f_n_part_sums[blockIdx.x] = diff;
		__threadfence();
		unsigned int value = atomicInc(count, gridDim.x);
		isLastBlockDone = (value == (gridDim.x - 1));´) m4_dnl use that reduction!
	__syncthreads();
	if (isLastBlockDone) {
		if (gridDim.x >  blockDim.x) {
			value = 0;

			for (int x=0; (gridDim.x % blokDim.x) == 0 ? x < (gridDim.x / blockDim.x) : x <= (gridDim.x / blockDim.x); x++)
				value += (gridDim.x % blockDim.x) == 0 || threadIdx.x * blockDim.x < gridDim.x ? f_n_part_sums[threadIdx.x * blockDim.x] : 0;
		}
		_CALL_BUTTERFLY_BLOCK_REDUCION(value, `*f_n = value;
		*count = 0;´) m4_dnl reduction across the partial sums
	}
}

__global__ void kernel_nabla_tilde_Gets_nabla_capped_with_rule( float* __restrict__ vec_nabla_tide_f, float* __restrict__ vec_nabla_f, float* __restrict__ vec_X) {
	_DEF_BLOCK_TOO_HIGH_THEADS_XY(`X´)
		return;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	vec_nabla_tilde_f[i] = (0 < vec_nabla_f[i] && 0 == vec_x[i]) ? 0 : vec_nabla_f[i];
}

__device__ _DEF_FFT_PRECISION(`R´) load_f_p_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ shared_Ptr) {
	m4_define(`_CALL_SPLIT_concatVar´, `int xPos = (offset / _DEF_concatVarSize) & _DEF_SIZE_concatVarSize;
	int yPos = offset & _DEF_concatVarSize;
	int patchNum = offset / _DEF_cpncatVarSize / _DEF_concatVarSize;´)
	m4_define(`_DEF_xPatchOffset´, m4_eval(((_DEF_storedSizeX ** 2) * (_DEF_NUM_PATCHES_Y + 1)) / 2))
	m4_define(`_DEF_yPatchOffset´, m4_eval(_DEF_storedSizeX / 2))
	m4_define(`_CALL_RESTRICT_WITH_PADDING´, `_CALL_SPLIT_concatVar
	m4_ifelse(`F´, `$1´, `int xPosStored, yPosStored;
	if (xPos <= _DEF_SIZE_HALF_F) { m4_dnl lower valid end
		xPosStored = xPos + _DEF_SIZE_HALF_F;
	} else if (_DEF_FFT_SIZE - _DEF_SIZE_HALF_F <= xPos) { m4_dnl upper valid end
		xPosStored = xPos + _DEF_SIZE_HALF_F - _DEF_FFT_SIZE;
	} else {
		$2
	}
	if (yPos <= _DEF_SIZE_HALF_F) {  m4_dnl lower valid end
		yPosStored = yPos + _DEF_SIZE_HALF_F;
	} else if (_DEF_FFT_SIZE - _DEF_SIZE_HALF_F <= yPos) { m4_dnl upper valid end
		yPosStored = yPos + _DEF_SIZE_HALF_F - _DEF_FFT_SIZE;
	} else {
		$2
	}´, `X´, `$1´, `int zero_space[4]; m4_dnl 0 <= x < 1 && 2 <= y < 3
	int xPatch = patchNum / _DEF_NUM_PATCHES_Y;
	int yPatch = patchNum - xPatch * _DEF_NUM_PATCHES_Y;
	if(0 == xPatch) { m4_dnl x = 0 border
		zero_space[0] = m4_eval(_DEF_SIZE_HALF_F + (_DEF_storedSizeX / 2));
		zero_space[1] = m4_ifelse(`Y´, `$2´, `_DEF_FFT_SIZE´, `X´, `$2´, `m4_eval(_DEF_SIZE_HALF_F + _DEF_storedSizeX)´);
	} else if (_DEF_NUM_PATCHES_Y - 1 == xPatch) { m4_dnl x = MAX border
		zero_space[0] = m4_ifelse(`Y´, `$2´, `0´, `X´, `$2´, `_DEF_SIZE_HALF_F´);
		zero_space[1] = m4_eval(_DEF_SIZE_HALF_F + (_DEF_storedSizeX / 2));
	} else { m4_dnl  no x border
		zero_space[0] = m4_ifelse(`Y´, `$2´, `0´, `X´, `$2´, `_DEF_SIZE_HALF_F´);
		zero_space[1] = m4_ifelse(`Y´, `$2´, `_DEF_FFT_SIZE´, `X´, `$2´, `m4_eval(_DEF_SIZE_HALF_F + _DEF_storedSizeX)´);
	}

	if (0 == yPatch) { m4_dnl y = 0 border
		zero_space[2] = m4_eval(_DEF_SIZE_HALF_F + (_DEF_storedSizeX /2));
		zero_space[3] = m4_ifelse(`Y´, `$2´, `_DEF_FFT_SIZE´, `X´, `$2´, `m4_eval(_DEF_SIZE_HALF_F + _DEF_storedSizeX)´);
	} else if (_DEF_NUM_PATCHES_X - 1 == yPatch) { m4_dnl y = MAX border
		zero_space[2] = m4_ifelse(`Y´, `$2´, `0´, `X´, `$2´, `_DEF_SIZE_HALF_F´);
		zero_space[3] = m4_eval(_DEF_SIZE_HALF_F + (_DEF_storedSizeX /2));
	} else { m4_dnl no y border
		zero_space[2] = m4_ifelse(`Y´, `$2´, `0´, `X´, `$2´, `_DEF_SIZE_HALF_F´);
		zero_space[3] = m4_ifelse(`Y´, `$2´, `_DEF_FFT_SIZE´, `X´, `$2´, `m4_eval(_DEF_SIZE_HALF_F + _DEF_storedSizeX)´);
	}

	int patcOffset = _DEF_xPatchOffset * xPatch + _DEF_yPatchOffset * yPatch;
	if (zero_space[0] <= xPos && xPos < zeroSpace[1] && zero_space[2] <= yPos && yPos < zero_space[3]) {
		int yPosStored = yPos - _DEF_SIZE_HALF_F;
		int xPosStored = xPos - _DEF_SIZE_HALF_F;
		$3
	}
	m4_dnl TODO: missing y and rest of it.
	m4_dnl TODO: complete for the zeroSpace[4] way in X and use $2, $3, etc. to select the right way, but only as many arguments as needed.
	´)´)
	_CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
	return ((_DEF_FFT_PRECISION(`R´)*) dataIn)[(_DEF_SIZE_F * _DEF_SIZE_F) *  patchNum + _DEF_SIZE_F * xPosStored + yPosStored] * _CALL_GEWICHTUNG(`xPos - _DEF_SIZE_HALF_F - (_DEF_storedSizeX/2)´, `yPos - _DEF_SIZE_HALF_F - (_DEF_storedSizeX/2)´);
}

__device__ void store_f_T_p_conj_fft_X(void* __restrict__ dataOut, size_t offset, _DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	((_DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = m4_ifelse(`float´, _DEF_FFT_PRECISION_TYPE, `cuConjf´, `double´, _DEF_FFT_PRECISION_TYPE, `cuConj´)(elment); m4_dnl TODO: insert right cmmand/
}

__device__ _DEF_FFT_PRECISION(`C´) load_F_X_m_F_X(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __retrict__ sharedPointer) {
	return m4_ifelse(`float´, _DEF_FFT_PRECISION_TYPE, `cuCmulf´)(((_DEF_FFT_PREC ISION(`C´)*) (dataIn))[offset], ((_DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);  m4_dnl TODO: include double precision as an option for the commplex multipliction.
}

__device__ void store_v_4_F_T_v_4_p_weight_half_v_1_X(void* __restrict__ dataOut, size_t offset, _DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	_CALL_RESTRICT_WITH_PADDING(`X´, `Y´, `element *= .5f * _CALL_GEWICHTUNG(`xPosStored - (_DEF_storedSizeX /2)´, `yPosStored - (_DEF_storedSizeX /2)´);
	atomicAdd(&((_DEF_FFT_PREISION(`R´)*) (dataOut))[patchOffset + xPosStored * _DEF_storedSizeX + yPosStored], element);
´)
}

__device__ _DEF_FFT_PRECISION(`R´) load_x_p_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrit__ sharedPtr) {
	_DEF_FFT_PRECISION(`R´) ret;
	_CALL_RESTRICT_WITH_PADDING(`X´, `Y´, `ret = ((_DEF_FFT_PRECISION(`R´)*) dataIn)[patchOffset + xPosStored * _DEF_storedSizeX + yPosStored];
		ret *= _CALL_GEWICHTUNG(`xPosStored´, `yPosStored´);
´) else
		ret = 0;
	return ret;
}

__device__ _DEF_FFT_PRECISION(`R´) load_f_X_1_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	_CALL_RESTRICT_WITH_PADDING(`F´, `return 0;´)
	return ((_DEF_FFT_PRECISION(`R´)*) dataIn)[(_DEF_SIZE_F * _DEF_SIZE_F) * patchNum + SIZE_F * xPosStoed + yPosStored];
}

__device__ _DEF_FFT_PRECISION(`R´) load_v_3_X_T_F(void* __restrict__ dataIn, size_t offset, void* __restrict__ callerInfo, void* __restrict__ sharedPtr) {
	_CALL_RESTRICT_WITH_PADDING(`X´, `X´, `return ((_DEF_FFT_PRECISION(`R´)*) (daaIn))[patchOffset + xPosStored * _DEF_storedSizeX + yPosStored];´) else
		return 0;
}

__device__ void store_f_X_y_p_v_1_F(void* __restrict__ dataOut, size_t offset, _DEF_FFT_PRECISION(`R´) element, void* __restrict__ callerInfo, void* __restrict__ sharedPointer) {
	_CALL_RESTRICT_WITH_PADDING(`X´, `X´, `atomicAdd(&((_DEF_FFT_PRECISION(`R´)*) (dataOut))[patchOffset + xPosStored * _DEF_storedSizeX + yPosStored], element * ((float) (1. / (_DEF_FFT_SIZE * _DEF_FFT_SIZE))))´)
}

__device__ void store_f_X_fft_m_x_F(void+ __restrict__ dataOut, size_t offset, _DEF_FFT_PRECISION(`C´) element, void* __restrict__ callerInfo, void* __retrict__ sharedPointer) {
	((_DEF_FFT_PRECISION(`C´)*) (dataOut))[offset] = cuCmulf(((_DEF_FFT_PRECISION(`C´)*) (datOut))[offset], ((_DEF_FFT_PRECISION(`C´)*) (callerInfo))[offset]);
}


