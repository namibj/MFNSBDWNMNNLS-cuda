/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * modified by Merlin Kramer.
 * to be better annotated.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cufftXt.h>
#include <assert.h>
#include <sm_30_intrinsics.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <driver_functions.h>
#define checkerboardSample
#ifdef checkerboardSample
#include "folded.c"
#include "unfolded.c"
#else
#include "00000001.c"
#include "00000002.c"
#endif
// assert() is only supported
// for devices of compute capability 2.0 and higher
//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
//#undef  assert
//#define assert(arg)
//#endif
#define NUM_TEST_PARALLEL_RUNS 1
static const int WORK_SIZE = 256;
static const int NUM_BITS_X_F_LOAD = 7;
static const int MASK_X_F_LOAD = 0x1 << (NUM_BITS_X_F_LOAD - 1);
static const int NUM_PATCHES_Y = 9;
static const int NUM_PATCHES_X = 9;
static const int SIZE_HALF_F = 64;
static const int SIZE_F = 2 * SIZE_HALF_F + 1;
static const uint storedSizeX = 128; //=r
static const int FFT_SIZE = storedSizeX + 2 * SIZE_HALF_F;
static const int SIZE_Y_X = (storedSizeX / 2) * (NUM_PATCHES_X + 1)
		- 2 * SIZE_HALF_F;
static const int M = 20; // lets say 20 for now, we can change it
static const float N_SOLL_F = 1e-40;
static const float N_SOLL_X = 1e-5;
static const float BETA_F = 0.5;
static const float SIGMA_F = 0.5;
static char output_f[(SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y];
static float spiked_f_h[NUM_PATCHES_X * NUM_PATCHES_Y * SIZE_F * SIZE_F];
struct store_f_X_T_1_informations {
	float* vec_f_o;
	int block_size;
	int block_num;
	float* vec_nabla_f_o;
	float beta;
	float alpha;
	float* nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d;
	float* abs_vec_nabla_f_part_sums;
	float* vec_delta_f;
	float* nabla_f_scalar_prod_delta_f_part_sums;
	float* abs_vec_delta_f_part_sums;
};
struct streamCallback_informations {
	struct store_f_X_T_1_informations* helper_struct_d;
	struct store_f_X_T_1_informations* helper_struct_h;
	int b;
	int finished;
	float* part_sums_var_h;
	float* delta_nabla_f_part_sums_h;
	float* nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h;
	float* f_n_h;
	float* f_o_h;
};
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_ERROR_CHECK CUDA_CHECK_RETURN(cudaGetLastError())
/*
 * This macro is used as a function for calculating the correct weighting
 *  factor at a given place, which is addressed from 0 to storedSizeX in
 *  coordinates which ignore the need for zeros as it is used in the OLA
 */
#define GEWICHTUNG(x,y)															\
		((1-abs((x)- (0.5f * (storedSizeX - 1))+0.5f)*(2.f/(storedSizeX+1)))*	\
		(1-abs((y)- (0.5f * (storedSizeX - 1))+0.5f)*(2.f/(storedSizeX+1))))
#define roundBlkSizeUp(count, blocksize) (((count) % (blocksize) ? (count) / (blocksize) : (count) / (blocksize) + 1))
__shared__ float part_Sums[32];
//__device__ unsigned int count = 0; // TODO: shift that thing into malloc-space, because we will have a problem as soon as more than one f gets optimized at the same time. (which is necessary to get good performance out of the algorithm)
__shared__ bool isLastBlockDone;

//  Just set count float's, starting from data, to zero
__global__ void kernel_set_float_zero(float* data, int lastBlockMax) {
	if (blockIdx.x != (gridDim.x - 1) || threadIdx.x < lastBlockMax)
		data[blockIdx.x * blockDim.x + threadIdx.x] = 0.f;

}

__global__ void
__launch_bounds__(1024, 2) kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr(
		float* __restrict__ v_3, float* __restrict__ y, float* __restrict__ y_k,
		float* __restrict__ f_n_part_sums, float* __restrict__ f_n,
		unsigned int* __restrict__ count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	float diff;
	float value;
	if (blockIdx.x == gridDim.x - 1
			&& threadIdx.x
					>= (((NUM_PATCHES_X + 1) * storedSizeX / 2 - 2 * SIZE_HALF_F)
							* ((NUM_PATCHES_Y + 1) * storedSizeX / 2
									- 2 * SIZE_HALF_F)
							- (gridDim.x - 1) * blockDim.x) - 1)
		diff = 0; // We are outside the image
	else {
		diff = y[index] - y_k[index];
		v_3[index] = diff;
	}
	// Reduction
	value = diff;
	int laneID = threadIdx.x & 0x1F;
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		part_Sums[threadIdx.x >> 5] = value;
	__syncthreads();
	if (threadIdx.x & ~0x1f == 0) { // Are we in the first warp of this block?

		value = part_Sums[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID) {
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			f_n_part_sums[blockIdx.x] = value;
			__threadfence();
			unsigned int value = atomicInc(count, gridDim.x);
			isLastBlockDone = (value == (gridDim.x - 1));
		}
	}
	__syncthreads();
	if (isLastBlockDone) {
		// We are the last block!
		// Reduction
		if (gridDim.x > blockDim.x) {
			value = 0;

			for (int x = 0;
					(gridDim.x % blockDim.x) == 0 ?
							x < (gridDim.x / blockDim.x) :
							x <= (gridDim.x / blockDim.x); x++)
				value +=
						(gridDim.x % blockDim.x) == 0
								|| threadIdx.x * blockDim.x < gridDim.x ?
								f_n_part_sums[threadIdx.x * blockDim.x] : 0;
		}
		int laneID = threadIdx.x & 0x1F;
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);

		if (0 == laneID)
			part_Sums[threadIdx.x >> 5] = value;
		__syncthreads();
		if (threadIdx.x & ~0x1f == 0) { // Are we in the first warp of this block?

			value = part_Sums[laneID];

			// Use XOR mode to perform butterfly reduction
			for (int i = 16; i >= 1; i /= 2)
				value += __shfl_xor(value, i, 32);
			if (0 == laneID) {
				// Are we the first thread in this block?
				// If so, then write the final sum into the correct location.
				*f_n = value;
				*count = 0;
			}

		}
	}
}

__global__ void kernel_nabla_tilde_Gets_nabla_capped_with_rule(
		float* __restrict__ vec_nabla_tilde_f, float* __restrict__ vec_nabla_f,
		float* __restrict__ vec_X) {
	if (blockIdx.x == gridDim.x - 1
			&& threadIdx.x
					>= ((NUM_PATCHES_X + 1) * storedSizeX / 2
							* (NUM_PATCHES_Y + 1) * storedSizeX / 2
							- (gridDim.x - 1) * blockDim.x) - 1)
		return;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	vec_nabla_tilde_f[i] =
			(0 < vec_nabla_f[i] && 0 == vec_X[i]) ? 0 : vec_nabla_f[i];
}

__device__ cufftReal load_f_p_X(void* dataIn, size_t offset, void* callerInfo,
		void* sharedPtr) {
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPosStored, yPosStored;
	if (xPos <= SIZE_HALF_F) { // lower valid end
		xPosStored = xPos + SIZE_HALF_F;
		assert(xPosStored < SIZE_F && xPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= xPos) { // upper valid end
		xPosStored = xPos + SIZE_HALF_F - FFT_SIZE;
		assert(xPosStored >= 0 && xPosStored < SIZE_HALF_F);
	} else {
		return 0;
	}
	if (yPos <= SIZE_HALF_F) { // lower valid end
		yPosStored = yPos + SIZE_HALF_F;
		assert(yPosStored < SIZE_F && yPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= yPos) { // upper valid end
		yPosStored = yPos + SIZE_HALF_F - FFT_SIZE;
		assert(yPosStored >= 0 && yPosStored < SIZE_HALF_F);
	} else {
		return 0;
	}
	return ((cufftReal*) dataIn)[(SIZE_F * SIZE_F) * patchNum
			+ SIZE_F * xPosStored + yPosStored]
			* GEWICHTUNG(xPos-SIZE_HALF_F - (storedSizeX/2), yPos-SIZE_HALF_F - (storedSizeX/2));
}

__device__ void store_f_T_p_conj_fft_X(void* dataOut, size_t offset,
		cufftComplex element, void* callerInfo, void* sharedPointer) {
	((cufftComplex*) (dataOut))[offset] = cuConjf(element); // The complex conjugate of the fft-output because we only need that.
}

__device__ cufftComplex load_F_X_m_F_X(void* dataIn, size_t offset,
		void* callerInfo, void* sharedPointer) {
	return cuCmulf(((cufftComplex*) (dataIn))[offset],
			((cufftComplex*) (callerInfo))[offset]); // The complex product of those two
}

__device__ void store_v_4_F_T_v_4_p_weight_half_v_1_X(void* dataOut,
		size_t offset, cufftReal element, void* callerInfo,
		void* sharedPointer) {
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPatch = patchNum / NUM_PATCHES_Y; //=i
	int yPatch = patchNum - xPatch * NUM_PATCHES_Y; //=j
	int zero_space[4]; // 0 <= x < 1 & 2 <= y < 3
	const int xPatchOffset = ((storedSizeX) ^ 2 * (NUM_PATCHES_Y + 1)) / 2; // Only go down half the patchsize because of the overlapping
	const int yPatchOffset = storedSizeX / 2;
	if (0 == xPatch) {
		// x = 0 border
		zero_space[0] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[1] = FFT_SIZE;
	} else if (NUM_PATCHES_Y - 1 == xPatch) {
		// x = MAX border
		zero_space[0] = 0;
		zero_space[1] = SIZE_HALF_F + (storedSizeX / 2);
	} else {
		// no x border
		zero_space[0] = 0;
		zero_space[1] = FFT_SIZE;
	}

	if (0 == yPatch) {
		// y = 0 border
		zero_space[2] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[3] = FFT_SIZE;
	} else if (NUM_PATCHES_X - 1 == yPatch) {
		// y = MAX border
		zero_space[2] = 0;
		zero_space[3] = SIZE_HALF_F + (storedSizeX / 2);
	} else {
		// no y border
		zero_space[2] = 0;
		zero_space[3] = FFT_SIZE;
	}

	int patchOffset = xPatchOffset * xPatch + yPatchOffset * yPatch;
	if (zero_space[0] <= xPos && xPos < zero_space[1] && zero_space[2] <= yPos
			&& yPos < zero_space[3]) {
		int xPosStored = xPos - SIZE_HALF_F;
		int yPosStored = yPos - SIZE_HALF_F;
		element *=
				.5f
						* GEWICHTUNG(xPosStored - (storedSizeX/2),yPosStored - (storedSizeX/2));
		atomicAdd(
				&((cufftReal*) (dataOut))[patchOffset + xPosStored * storedSizeX
						+ yPosStored], element);
	}
}

__device__ cufftReal load_x_p_F(void* dataIn, size_t offset, void* callerInfo,
		void* sharedPtr) {
	cufftReal ret;
	assert(false);
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPatch = patchNum / NUM_PATCHES_Y; //=i
	int yPatch = patchNum - xPatch * NUM_PATCHES_Y; //=j
	int zero_space[4]; // 0 <= x < 1 & 2 <= y < 3
	const int xPatchOffset = ((storedSizeX) ^ 2 * (NUM_PATCHES_Y + 1)) / 2; // Only go down half the patchsize because of the overlapping
	const int yPatchOffset = storedSizeX / 2;
	if (0 == xPatch) { // x = 0 border
		zero_space[0] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[1] = SIZE_HALF_F + storedSizeX;
	} else if (NUM_PATCHES_Y - 1 == xPatch) { // x = MAX border
		zero_space[0] = SIZE_HALF_F;
		zero_space[1] = SIZE_HALF_F + (storedSizeX / 2);
	} else { // no x border
		zero_space[0] = SIZE_HALF_F;
		zero_space[1] = SIZE_HALF_F + storedSizeX;
	}
	if (0 == yPatch) { // y = 0 border
		zero_space[2] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[3] = SIZE_HALF_F + storedSizeX;
	} else if (NUM_PATCHES_X - 1 == yPatch) { // y = MAX border
		zero_space[2] = SIZE_HALF_F;
		zero_space[3] = SIZE_HALF_F + (storedSizeX / 2);
	} else { // no y border
		zero_space[2] = SIZE_HALF_F;
		zero_space[3] = SIZE_HALF_F + storedSizeX;
	}
	int patchOffset = xPatchOffset * xPatch + yPatchOffset * yPatch;
	if (zero_space[0] <= xPos && xPos < zero_space[1] // x not in  the OLA zero-space
	&& zero_space[2] <= yPos && yPos < zero_space[3] // y not in the OLA zero-space
			) {

		int xPosStored = xPos - SIZE_HALF_F;
		int yPosStored = yPos - SIZE_HALF_F;

		ret = ((cufftReal*) dataIn)[patchOffset + xPosStored * storedSizeX
				+ yPosStored]; //retrieving the value from its correct locationcudaStreamWaitEvent(stream, 2, 0);
		ret *= GEWICHTUNG(xPosStored, yPosStored); //TODO: check if all uses of GEWICHTUNG(x,y) are correct as they must not supply negative values for x and / or y (0<=x<s
		// 1-(Abs[x - 0.5*(s - 1)]+0.5)/(0.5*(s+1)+0.5)
	} else
		ret = 0;
	return ret;
}

__device__ cufftReal load_f_X_1_F(void* dataIn, size_t offset, void* callerInfo,
		void* sharedPtr) {
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPosStored, yPosStored;
	if (xPos <= SIZE_HALF_F) { // lower valid end
		xPosStored = xPos + SIZE_HALF_F;
		assert(xPosStored < SIZE_F && xPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= xPos) { // upper valid end
		xPosStored = xPos + SIZE_HALF_F - FFT_SIZE;
		assert(xPosStored >= 0 && xPosStored < SIZE_HALF_F);
	} else {
		return 0;
	}
	if (yPos <= SIZE_HALF_F) { // lower valid end
		yPosStored = yPos + SIZE_HALF_F;
		assert(yPosStored < SIZE_F && yPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= yPos) { // upper valid end
		yPosStored = yPos + SIZE_HALF_F - FFT_SIZE;
		assert(yPosStored >= 0 && yPosStored < SIZE_HALF_F);
	} else {
		return 0;
	}
	assert(
			((SIZE_F * SIZE_F) * patchNum + SIZE_F * xPosStored + yPosStored)
					< SIZE_F * SIZE_F * NUM_PATCHES_X * NUM_PATCHES_Y);
	return ((cufftReal*) dataIn)[(SIZE_F * SIZE_F) * patchNum
			+ SIZE_F * xPosStored + yPosStored];
}

__device__ cufftReal load_v_3_X_T_F(void* dataIn, size_t offset,
		void* callerInfo, void* sharedPtr) {
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPatch = patchNum / NUM_PATCHES_Y; //=i
	int yPatch = patchNum - xPatch * NUM_PATCHES_Y; //=j
	int zero_space[4]; // 0 <= x < 1 & 2 <= y < 3
	const int xPatchOffset = ((storedSizeX) ^ 2 * (NUM_PATCHES_Y + 1)) / 2; // Only go down half the patchsize because of the overlapping
	const int yPatchOffset = storedSizeX / 2;
	if (0 == xPatch) {
		// x = 0 border
		zero_space[0] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[1] = FFT_SIZE;
	} else if (NUM_PATCHES_Y - 1 == xPatch) {
		// x = MAX border
		zero_space[0] = 0;
		zero_space[1] = SIZE_HALF_F + (storedSizeX / 2);
	} else {
		// no x border
		zero_space[0] = 0;
		zero_space[1] = FFT_SIZE;
	}

	if (0 == yPatch) {
		// y = 0 border
		zero_space[2] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[3] = FFT_SIZE;
	} else if (NUM_PATCHES_X - 1 == yPatch) {
		// y = MAX border
		zero_space[2] = 0;
		zero_space[3] = SIZE_HALF_F + (storedSizeX / 2);
	} else {
		// no y border
		zero_space[2] = 0;
		zero_space[3] = FFT_SIZE;
	}

	int patchOffset = xPatchOffset * xPatch + yPatchOffset * yPatch;
	if (zero_space[0] <= xPos && xPos < zero_space[1] && zero_space[2] <= yPos
			&& yPos < zero_space[3]) {
		int xPosStored = xPos - SIZE_HALF_F;
		int yPosStored = yPos - SIZE_HALF_F;
		return ((cufftReal*) (dataIn))[patchOffset + xPosStored * storedSizeX
				+ yPosStored];
	} else
		return 0;
}

__device__ void store_f_X_y_p_v_1_F(void* dataOut, size_t offset,
		cufftReal element, void* callerInfo, void* sharedPointer) {
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPatch = patchNum / NUM_PATCHES_Y; //=i
	int yPatch = patchNum - xPatch * NUM_PATCHES_Y; //=j
	int zero_space[4]; // 0 <= x < 1 & 2 <= y < 3
	const int xPatchOffset = ((storedSizeX) ^ 2 * (NUM_PATCHES_Y + 1)) / 2; // Only go down half the patchsize because of the overlapping
	const int yPatchOffset = storedSizeX / 2;
	if (0 == xPatch) {
		// x = 0 border
		zero_space[0] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[1] = FFT_SIZE;
	} else if (NUM_PATCHES_Y - 1 == xPatch) {
		// x = MAX border
		zero_space[0] = 0;
		zero_space[1] = SIZE_HALF_F + (storedSizeX / 2);
	} else {
		// no x border
		zero_space[0] = 0;
		zero_space[1] = FFT_SIZE;
	}

	if (0 == yPatch) {
		// y = 0 border
		zero_space[2] = SIZE_HALF_F + (storedSizeX / 2);
		zero_space[3] = FFT_SIZE;
	} else if (NUM_PATCHES_X - 1 == yPatch) {
		// y = MAX border
		zero_space[2] = 0;
		zero_space[3] = SIZE_HALF_F + (storedSizeX / 2);
	} else {
		// no y border
		zero_space[2] = 0;
		zero_space[3] = FFT_SIZE;
	}

	// Do not forget to normalize the calculations
	element *= (float) (1. / (FFT_SIZE * FFT_SIZE));

	int patchOffset = xPatchOffset * xPatch + yPatchOffset * yPatch;
	if (zero_space[0] <= xPos && xPos < zero_space[1] && zero_space[2] <= yPos
			&& yPos < zero_space[3]) {
		int xPosStored = xPos - SIZE_HALF_F;
		int yPosStored = yPos - SIZE_HALF_F;
		atomicAdd(
				&((cufftReal*) (dataOut))[patchOffset + xPosStored * storedSizeX
						+ yPosStored], element);
	}
}

__device__ void store_f_X_fft_m_x_F(void* dataOut, size_t offset,
		cufftComplex element, void* callerInfo, void* sharedPointer) {
	assert(offset < FFT_SIZE * (FFT_SIZE / 2 + 1));
	((cufftComplex*) (dataOut))[offset] = cuCmulf(
			((cufftComplex*) (dataOut))[offset],
			((cufftComplex*) (callerInfo))[offset]); // The corresponding X' value
}

__device__ void store_f_X_T_fft_m_x_F(void* dataOut, size_t offset,
		cufftComplex element, void* callerInfo, void* sharedPointer) {
	((cufftComplex*) (dataOut))[offset] = cuCmulf(
			((cufftComplex*) (dataOut))[offset],
			cuConjf(((cufftComplex*) (callerInfo))[offset])); // The complex conjugate of the corresponding X' value
}

__device__ void store_f_X_T_1_nabla_tilde_f_even_b_F(void *dataOut,
		size_t offset, cufftReal element, void* callerInfo,
		void* sharedPointer) {
	/*
	 * \nabla\tilde{f} \gets \nabla[f} .* {\nalba f\lessequal 0 \or X \notequal 0}
	 * Update
	 * ||\nabla\tilde[f}||
	 *  <\nabla f_o,\ (x_o - f)>
	 * 		Skalarprodukt von zu ladendem alten \nabla\tilde{f} und der Differenz von geladenem f (x_o) und neuem f (f)
	 * 		Im Anschluss an die FFT sind die Einzelsummen noch zu summieren
	 */

	struct store_f_X_T_1_informations (*inform_struct) =
			(store_f_X_T_1_informations*) (callerInfo);
	float nabla_tilde_f = 0;
	float value;
	float f;
	float *vec_f_o = inform_struct->vec_f_o;
	float *vec_f = vec_f_o;
	float *vec_nabla_tilde_f_o = inform_struct->vec_nabla_f_o;
// Usual index computations
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPosStored, yPosStored;
	bool isF = true;
	if (xPos <= SIZE_HALF_F) { // lower valid end
		xPosStored = xPos + SIZE_HALF_F;
		assert(xPosStored < SIZE_F && xPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= xPos) { // upper valid end
		xPosStored = xPos + SIZE_HALF_F - FFT_SIZE;
		assert(xPosStored >= 0 && xPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		isF = false;
		goto sumItUp;
		// Outside of f-range
	};
	if (yPos <= SIZE_HALF_F) { // lower valid end
		yPosStored = yPos + SIZE_HALF_F;
		assert(yPosStored < SIZE_F && yPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= yPos) { // upper valid end
		yPosStored = yPos + SIZE_HALF_F - FFT_SIZE;
		assert(yPosStored >= 0 && yPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		isF = false;
		goto sumItUp;
		// Outside of f-range
	};
	int index = (SIZE_F * SIZE_F) * patchNum + SIZE_F * xPosStored + yPosStored;

// Directly create \nabla\tilde{f} and skip those who are classified to be zeroed
	if (element > 0 && 0 == vec_f_o[index])
		nabla_tilde_f = 0;
	else
		// Do not forget to normalize the calculations
		nabla_tilde_f = ((float) .5 / (FFT_SIZE * FFT_SIZE)) * element;
// Seed starting value

	f = vec_f_o[index]
			- inform_struct->alpha * inform_struct->beta * nabla_tilde_f;
	value = vec_nabla_tilde_f_o[index] * (vec_f_o[index] - vec_f[index]);

	sumItUp: // If we do not have a real value for f here, just continue to get the summing correct
	int laneID = threadIdx.x & 0x1f;

// Use XOR mode to perform butterfly reduction
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		((float*) sharedPointer)[(threadIdx.z * blockDim.z
				+ threadIdx.y * blockDim.y + threadIdx.x * blockDim.x) >> 5] =
				value;
	__syncthreads();
	if ((threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) & ~0x1f == 0) { // Are we in the first warp of this block?

		value = ((float*) sharedPointer)[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID)
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			inform_struct->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d[gridDim.x
					* gridDim.y * blockIdx.z + gridDim.x * blockIdx.y
					+ blockIdx.x] = value;
	}

	/*
	 * Sum up the squared value of nabla_tilde_f
	 */
	// Seed starting value
	value = nabla_tilde_f * nabla_tilde_f;

	// Use XOR mode to perform butterfly reduction
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		((float*) sharedPointer)[(threadIdx.z * blockDim.z
				+ threadIdx.y * blockDim.y + threadIdx.x * blockDim.x) >> 5] =
				value;
	__syncthreads();
	if ((threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) & ~0x1f == 0) { // Are we in the first warp of this block?

		value = ((float*) sharedPointer)[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID) {
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			inform_struct->abs_vec_nabla_f_part_sums[gridDim.x * gridDim.y
					* blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x] = value;
			inform_struct->block_num = gridDim.x * gridDim.y * gridDim.z;
			inform_struct->block_size = blockDim.x * blockDim.y * blockDim.z;
		}
	}

	// Update only if it is a real f
	if (isF)
		vec_f_o[index] = f;
}

__device__ void store_f_X_T_1_nabla_tilde_f_uneven_b_F(void* dataOut,
		size_t offset, cufftReal element, void* callerInfo,
		void* sharedPointer) {
	/*
	 * \nabla\tilde{f} \gets \nabla[f} .* {\nalba f\lessequal 0 \or X \notequal 0}
	 * Update
	 *  <\nabla f_o,\ (x_o - f)>
	 * 		Skalarprodukt von zu ladendem alten \nabla\tilde{f} und der Differenz von geladenem f (x_o) und neuem f (f)
	 * 		Im Anschluss an die FFT sind die Einzelsummen noch zu summieren
	 */

	struct store_f_X_T_1_informations (*inform_struct) =
			(store_f_X_T_1_informations*) (callerInfo);
	float nabla_tilde_f = 0;
	float value;
	float f;
	float *vec_f_o = inform_struct->vec_f_o;
	float *vec_f = vec_f_o;
	float *vec_nabla_tilde_f_o = inform_struct->vec_nabla_f_o;
	// Usual index computations
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPosStored, yPosStored;
	bool isF = true;
	if (xPos <= SIZE_HALF_F) { // lower valid end
		xPosStored = xPos + SIZE_HALF_F;
		assert(xPosStored < SIZE_F && xPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= xPos) { // upper valid end
		xPosStored = xPos + SIZE_HALF_F - FFT_SIZE;
		assert(xPosStored >= 0 && xPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		isF = false;
		goto sumItUp;
		// Outside of f-range
	};
	if (yPos <= SIZE_HALF_F) { // lower valid end
		yPosStored = yPos + SIZE_HALF_F;
		assert(yPosStored < SIZE_F && yPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= yPos) { // upper valid end
		yPosStored = yPos + SIZE_HALF_F - FFT_SIZE;
		assert(yPosStored >= 0 && yPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		isF = false;
		goto sumItUp;
		// Outside of f-range
	};
	int index = (SIZE_F * SIZE_F) * patchNum + SIZE_F * xPosStored + yPosStored;

	// Directly create \nabla\tilde{f} and skip those who are classified to be zeroed
	if (element > 0 && 0 == vec_f_o[index])
		nabla_tilde_f = 0;
	else
		// Do not forget to normalize the calculations
		nabla_tilde_f = ((float) .5 / (FFT_SIZE * FFT_SIZE)) * element;
	// Seed starting value

	f = vec_f_o[index]
			- inform_struct->alpha * inform_struct->beta * nabla_tilde_f;
	value = vec_nabla_tilde_f_o[index] * (vec_f_o[index] - vec_f[index]);

	sumItUp: // If we do not have a real value for f here, just continue to get the summing correct
	int laneID = threadIdx.x & 0x1f;

	// Use XOR mode to perform butterfly reduction
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		((float*) sharedPointer)[(threadIdx.z * blockDim.z
				+ threadIdx.y * blockDim.y + threadIdx.x * blockDim.x) >> 5] =
				value;
	__syncthreads();
	if ((threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) & ~0x1f == 0) { // Are we in the first warp of this block?

		value = ((float*) sharedPointer)[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID) {
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			inform_struct->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d[gridDim.x
					* gridDim.y * blockIdx.z + gridDim.x * blockIdx.y
					+ blockIdx.x] = value;
			inform_struct->block_num = gridDim.x * gridDim.y * gridDim.z;
			inform_struct->block_size = blockDim.x * blockDim.y * blockDim.z;
		}
	}

	// Update only if it is a real f
	if (isF)
		vec_f_o[index] = f;
}

__device__ void store_f_X_T_2_delta_tilde_f_even_b_F(void* dataOut,
		size_t offset, cufftReal element, void* callerInfo,
		void* sharedPointer) {
	/*
	 * \delta\tilde{f}
	 *  <\nabla\tilde{f},\ \delta\tilde{f}>
	 * 		Skalarprodukt von zu ladendem alten \nabla\tilde{f} und der Differenz von geladenem f (x_o) und neuem f (f)
	 * 		Im Anschluss an die FFT sind die Einzelsummen noch zu summieren
	 */

	struct store_f_X_T_1_informations (*inform_struct) =
			(store_f_X_T_1_informations*) (callerInfo);
	float value;
	// Usual index computations
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPosStored, yPosStored;
	if (xPos <= SIZE_HALF_F) { // lower valid end
		xPosStored = xPos + SIZE_HALF_F;
		assert(xPosStored < SIZE_F && xPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= xPos) { // upper valid end
		xPosStored = xPos + SIZE_HALF_F - FFT_SIZE;
		assert(xPosStored >= 0 && xPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		goto sumItUp;
		// Outside of f-range
	};
	if (yPos <= SIZE_HALF_F) { // lower valid end
		yPosStored = yPos + SIZE_HALF_F;
		assert(yPosStored < SIZE_F && yPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= yPos) { // upper valid end
		yPosStored = yPos + SIZE_HALF_F - FFT_SIZE;
		assert(yPosStored >= 0 && yPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		goto sumItUp;
		// Outside of f-range
	};
	int index = (SIZE_F * SIZE_F) * patchNum + SIZE_F * xPosStored + yPosStored;
	// We do not need to save \delta\tilde{f}, because it is only used in the computations we do here on it.

	// Seed starting value = \nabla\tilde{f} * \delta\tilde{f}
	// And do not forget to normalize the calculations
	value = inform_struct->vec_nabla_f_o[index] * element
			* ((float) 1. / (FFT_SIZE * FFT_SIZE));

	sumItUp: // If we do not have a real value for f here, just continue to get the summing correct
	int laneID = threadIdx.x & 0x1f;

	// Use XOR mode to perform butterfly reduction
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		((float*) sharedPointer)[(threadIdx.z * blockDim.z
				+ threadIdx.y * blockDim.y + threadIdx.x * blockDim.x) >> 5] =
				value;
	__syncthreads();
	if ((threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) & ~0x1f == 0) { // Are we in the first warp of this block?

		value = ((float*) sharedPointer)[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID) {
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			inform_struct->nabla_f_scalar_prod_delta_f_part_sums[gridDim.x
					* gridDim.y * blockIdx.z + gridDim.x * blockIdx.y
					+ blockIdx.x] = value;
			inform_struct->block_num = gridDim.x * gridDim.y * gridDim.z;
			inform_struct->block_size = blockDim.x * blockDim.y * blockDim.z;
		}
	}
}

__device__ void store_f_X_T_2_delta_tilde_f_uneven_b_F(void* dataOut,
		size_t offset, cufftReal element, void* callerInfo,
		void* sharedPointer) {
	/*
	 * \delta\tilde{f}
	 * <\nabla\tilde{f},\ \delta\tilde{f}>
	 * ||\delta\tilde{f}||
	 */

	struct store_f_X_T_1_informations (*inform_struct) =
			(store_f_X_T_1_informations*) (callerInfo);
	float value;
	float *vec_f_o = inform_struct->vec_f_o;
	// Usual index computations
	int xPos = (offset >> NUM_BITS_X_F_LOAD) && MASK_X_F_LOAD; //=k'
	int yPos = offset && MASK_X_F_LOAD; //=l'
	int patchNum = offset >> (NUM_BITS_X_F_LOAD * 2);
	int xPosStored, yPosStored;
	if (xPos <= SIZE_HALF_F) { // lower valid end
		xPosStored = xPos + SIZE_HALF_F;
		assert(xPosStored < SIZE_F && xPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= xPos) { // upper valid end
		xPosStored = xPos + SIZE_HALF_F - FFT_SIZE;
		assert(xPosStored >= 0 && xPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		goto sumItUp;
		// Outside of f-range
	};
	if (yPos <= SIZE_HALF_F) { // lower valid end
		yPosStored = yPos + SIZE_HALF_F;
		assert(yPosStored < SIZE_F && yPosStored >= SIZE_HALF_F);
	} else if (FFT_SIZE - SIZE_HALF_F <= yPos) { // upper valid end
		yPosStored = yPos + SIZE_HALF_F - FFT_SIZE;
		assert(yPosStored >= 0 && yPosStored < SIZE_HALF_F);
	} else {
		value = 0;
		goto sumItUp;
		// Outside of f-range
	};
	int index = (SIZE_F * SIZE_F) * patchNum + SIZE_F * xPosStored + yPosStored;
	// We do not need to save \delta\tilde{f}, because it is only used in the computations we do here on it.

	// Seed starting value = \nabla\tilde{f} * \delta\tilde{f}
	// And do not forget to normalize the calculations
	value = inform_struct->vec_nabla_f_o[index] * element
			* ((float) 1. / (FFT_SIZE * FFT_SIZE));

	sumItUp: // If we do not have a real value for f here, just continue to get the summing correct
	int laneID = threadIdx.x & 0x1f;

	// Use XOR mode to perform butterfly reduction
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		((float*) sharedPointer)[(threadIdx.z * blockDim.z
				+ threadIdx.y * blockDim.y + threadIdx.x * blockDim.x) >> 5] =
				value;
	__syncthreads();
	if ((threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) & ~0x1f == 0) { // Are we in the first warp of this block?

		value = ((float*) sharedPointer)[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID)
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			inform_struct->nabla_f_scalar_prod_delta_f_part_sums[gridDim.x
					* gridDim.y * blockIdx.z + gridDim.x * blockIdx.y
					+ blockIdx.x] = value;
	}
// Now on to ||\delta\tilde{f}||
	value = element * element;
// Use XOR mode to perform butterfly reduction
	for (int i = 16; i >= 1; i /= 2)
		value += __shfl_xor(value, i, 32);

	if (0 == laneID)
		((float*) sharedPointer)[(threadIdx.z * blockDim.z
				+ threadIdx.y * blockDim.y + threadIdx.x * blockDim.x) >> 5] =
				value;
	__syncthreads();
	if ((threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x
			+ threadIdx.x) & ~0x1f == 0) { // Are we in the first warp of this block?

		value = ((float*) sharedPointer)[laneID];

		// Use XOR mode to perform butterfly reduction
		for (int i = 16; i >= 1; i /= 2)
			value += __shfl_xor(value, i, 32);
		if (0 == laneID) {
			// Are we the first thread in this block?
			// If so, then write the sum over this block into the correct location in the part_sums-array.
			inform_struct->abs_vec_delta_f_part_sums[gridDim.x * gridDim.y
					* blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x] = value;
			inform_struct->block_num = gridDim.x * gridDim.y * gridDim.z;
			inform_struct->block_size = blockDim.x * blockDim.y * blockDim.z;
		}
	}
}
//__device__ void (*)() tmp_pointer = (void(*)()) ;
__device__ cufftCallbackLoadR _d_load_x_p_F = load_x_p_F;
__device__ cufftCallbackStoreR _d_store_f_X_y_p_v_1_F = store_f_X_y_p_v_1_F;
__device__ cufftCallbackStoreC _d_store_f_X_fft_m_x_F = store_f_X_fft_m_x_F;
__device__ cufftCallbackStoreC _d_store_f_X_T_fft_m_x_F = store_f_X_T_fft_m_x_F;
__device__ cufftCallbackLoadR _d_load_f_X_1_F = load_f_X_1_F;
__device__ cufftCallbackLoadR _d_load_v_3_X_T_F = load_v_3_X_T_F;
__device__ cufftCallbackStoreR _d_store_f_X_T_1_nabla_tilde_f_even_b_F =
		store_f_X_T_1_nabla_tilde_f_even_b_F;
__device__ cufftCallbackStoreR _d_store_f_X_T_1_nabla_tilde_f_uneven_b_F =
		store_f_X_T_1_nabla_tilde_f_uneven_b_F;
__device__ cufftCallbackStoreR _d_store_f_X_T_2_delta_tilde_f_even_b_F =
		store_f_X_T_2_delta_tilde_f_even_b_F;
__device__ cufftCallbackStoreR _d_store_f_X_T_2_delta_tilde_f_uneven_b_F =
		store_f_X_T_2_delta_tilde_f_uneven_b_F;
cufftCallbackLoadR _h_load_x_p_F;
cufftCallbackStoreR _h_store_f_X_y_p_v_1_F;
cufftCallbackStoreC _h_store_f_X_fft_m_x_F;
cufftCallbackStoreC _h_store_f_X_T_fft_m_x_F;
cufftCallbackStoreC _h_load_f_X_1_F;
cufftCallbackLoadR _h_load_v_3_X_T_F;
cufftCallbackStoreR _h_store_f_X_T_1_nabla_tilde_f_even_b_F;
cufftCallbackStoreR _h_store_f_X_T_1_nabla_tilde_f_uneven_b_F;
cufftCallbackStoreR _h_store_f_X_T_2_delta_tilde_f_even_b_F;
cufftCallbackStoreR _h_store_f_X_T_2_delta_tilde_f_uneven_b_F;
void getCallbacks() {
	cudaMemcpyFromSymbol(&_h_load_x_p_F, _d_load_x_p_F, sizeof(_h_load_x_p_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_y_p_v_1_F, _d_store_f_X_y_p_v_1_F,
			sizeof(_h_store_f_X_y_p_v_1_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_fft_m_x_F, _d_store_f_X_fft_m_x_F,
			sizeof(_h_store_f_X_fft_m_x_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_T_fft_m_x_F, _d_store_f_X_T_fft_m_x_F,
			sizeof(_h_store_f_X_T_fft_m_x_F));
	cudaMemcpyFromSymbol(&_h_load_f_X_1_F, _d_load_f_X_1_F,
			sizeof(_h_load_f_X_1_F));
	cudaMemcpyFromSymbol(&_h_load_v_3_X_T_F, _d_load_v_3_X_T_F,
			sizeof(_h_load_v_3_X_T_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_T_1_nabla_tilde_f_even_b_F,
			_d_store_f_X_T_1_nabla_tilde_f_even_b_F,
			sizeof(_h_store_f_X_T_1_nabla_tilde_f_even_b_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_T_1_nabla_tilde_f_uneven_b_F,
			_d_store_f_X_T_1_nabla_tilde_f_uneven_b_F,
			sizeof(_h_store_f_X_T_1_nabla_tilde_f_uneven_b_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_T_2_delta_tilde_f_even_b_F,
			_d_store_f_X_T_2_delta_tilde_f_even_b_F,
			sizeof(_h_store_f_X_T_2_delta_tilde_f_even_b_F));
	cudaMemcpyFromSymbol(&_h_store_f_X_T_2_delta_tilde_f_uneven_b_F,
			_d_store_f_X_T_2_delta_tilde_f_uneven_b_F,
			sizeof(_h_store_f_X_T_2_delta_tilde_f_uneven_b_F));
}

int setFloatDeviceZero(float* data, size_t count, int blocksize,
		cudaStream_t stream) {
	// Round numBlocks up
	kernel_set_float_zero<<<roundBlkSizeUp(count,blocksize), blocksize, 0,
			stream>>>(data, count % blocksize);
	return 0;
}

void optimizeFcallback(cudaStream_t stream, cudaError_t status,
		void* userData) {
	struct streamCallback_informations *informations =
			((struct streamCallback_informations*) userData);
	//printf("");
	if (informations->b % 2 == 0) { // even
		float abs_nabla_f = 0;
		float delta_nabla_f = 0;
		if (*(informations->f_n_h) < N_SOLL_F) {
			// finished with this optimization
			informations->finished = true;  // signal finish to the loop
			return;
		}
		for (int i = 0; i < informations->helper_struct_h->block_num; i++) {
			abs_nabla_f += informations->part_sums_var_h[i];
			delta_nabla_f += informations->delta_nabla_f_part_sums_h[i];
		}
		informations->helper_struct_h->alpha = abs_nabla_f / delta_nabla_f;
		//float n_a = informations.helper_struct_h->
	} else {
		float abs_delta_f = 0;
		float delta_nabla_f = 0;
		for (int i = 0; i < informations->helper_struct_h->block_num; i++) {
			abs_delta_f += informations->part_sums_var_h[i];
			delta_nabla_f += informations->delta_nabla_f_part_sums_h[i];
		}
		informations->helper_struct_h->alpha = delta_nabla_f / abs_delta_f;
	}
	float complicatetSums = 0;
	for (int i = 0; i < informations->helper_struct_h->block_num; i++)
		complicatetSums +=
				(informations->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h)[i];
	if (informations->f_o_h - informations->f_n_h <= SIGMA_F * complicatetSums)
		informations->helper_struct_h->beta *= BETA_F;
	informations->f_o_h = informations->f_n_h;
}

// call with "_F.pgm"
void output_size_F(float* f_d, char* fname) {

	cudaDeviceSynchronize();
	float* f_h;
	cudaMallocHost(&f_h,
			sizeof(float) * NUM_PATCHES_X * NUM_PATCHES_Y * SIZE_F * SIZE_F);
	cudaMemcpy(f_h, f_d,
			sizeof(float) * NUM_PATCHES_X * NUM_PATCHES_Y * SIZE_F * SIZE_F,
			cudaMemcpyDeviceToHost);
	// Open the PGM output file
	FILE* file = fopen(fname, "wb");

	// Write the file header
	fprintf(file, "P5\n"
			"%d %d\n"
			"255\n", (SIZE_F * NUM_PATCHES_Y), (SIZE_F * NUM_PATCHES_X));

	double max = 0;
	for (int x = 0; x < NUM_PATCHES_X * SIZE_F; x++)
		for (int y = 0; y < NUM_PATCHES_Y * SIZE_F; y++)
			max = f_h[x * (NUM_PATCHES_Y * SIZE_F) + y] > max ?
					f_h[x * (NUM_PATCHES_Y * SIZE_F) + y] : max;
	// Write the content: remember, data is saved in x-major order, both the patches themself and the data inside of the patches
	for (int xPatch = 0; xPatch < (NUM_PATCHES_X); xPatch++)
		for (int xPos = 0; xPos < SIZE_F; xPos++)
			for (int yPatch = 0; yPatch < (NUM_PATCHES_Y); yPatch++)
				for (int yPos = 0; yPos < SIZE_F; yPos++)
					fputc(
							((char) ((uint) (f_h[(xPatch * NUM_PATCHES_Y
									+ yPatch) * SIZE_F * SIZE_F + xPos * SIZE_F
									+ yPos] * (255 / max)))), file);

	fclose(file);
	cudaFreeHost(f_h);
}

// call with "_X.pgm"
void output_size_X(float* x_d, char* fname) {

	cudaDeviceSynchronize();
	float* x_h;
	cudaMallocHost(&x_h, sizeof(float) * 512 * 512);
	cudaMemcpy(x_h, x_d, sizeof(float) * 512 * 512, cudaMemcpyDeviceToHost);
	// Open the PGM output file
	FILE* file = fopen(fname, "wb");

	printf("Writing %s_X.pgm", fname);
	// Write the file header
	fprintf(file, "P5\n"
			"%d %d\n"
			"255\n", 512, 512);

	double max = 0;
	for (int x = 0; x < 512; x++)
		for (int y = 0; y < 512; y++)
			max = x_h[x * 512 + y] > max ? x_h[x * 512 + y] : max;
	// Write the content: remember, data is saved in x-major order, both the patches themself and the data inside of the patches
	for (int x = 0; x < 512; x++)
		for (int y = 0; y < 512; y++)
			fputc(((char) ((uint) (x_h[x * 512 + y] * (255 / max)))), file);

	fclose(file);
	cudaFreeHost(x_h);
}

int optimizeF(float* f_h, float* y_k_h, float* x_h, cudaStream_t stream) {
	int dev;
	float* f_d = NULL;
	float* x_d = NULL;
	float* v_3_d = NULL;
	float* y_d = NULL;
	float* y_k_d = NULL;
	float* f_n_part_sums_d = NULL;
	float* f_n_d = NULL;
	float* f_n_h = NULL;
	char* str = (char*) malloc(80);
	cudaMallocHost(&f_n_h, sizeof(float));
	float* delta_nabla_f_part_sums_h;
	float* part_sums_var_h;
	unsigned int* count_d = NULL;
	cufftComplex* v_tmp_cmplx_d = NULL;
	cufftComplex *x_p_d = NULL;
	struct store_f_X_T_1_informations* helper_struct_d = NULL;
	struct store_f_X_T_1_informations* helper_struct_h;
	helper_struct_h = (struct store_f_X_T_1_informations*) malloc(
			sizeof(store_f_X_T_1_informations));
	struct streamCallback_informations* streamCallback =
			(struct streamCallback_informations*) malloc(
					sizeof(struct streamCallback_informations));
//	helper_struct_h = (struct store_f_X_T_1_informations*) malloc(
//			sizeof(store_f_X_T_1_informations));
/// Space for $f$
	cudaMalloc((void**) &f_d,
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
/// Space for $f_o$
	cudaMalloc((void**) &(helper_struct_h->vec_f_o),
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
/// Space for $\nabla f_o$
	cudaMalloc((void**) &(helper_struct_h->vec_nabla_f_o),
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
/// Space for pre-calculated $X'$
	cudaMalloc((void**) &x_p_d,
			sizeof(cufftComplex) * FFT_SIZE * (FFT_SIZE / 2 + 1) * NUM_PATCHES_X
					* NUM_PATCHES_Y);
/// Space for $X$ (only to then calculate $X'$ out of it)
	cudaMalloc((void**) &x_d,
			sizeof(float) * (storedSizeX / 2) * (NUM_PATCHES_X + 1)
					* (storedSizeX / 2) * (NUM_PATCHES_Y + 1));
/// Space for $v_3$ // Couldn't we instead just reuse the $X$-space, because we don't need it anymore after computing $X'$ ?
	cudaMalloc((void**) &v_3_d,
			sizeof(float) * (storedSizeX / 2) * (NUM_PATCHES_X + 1)
					* (storedSizeX / 2) * (NUM_PATCHES_Y + 1));
/// Space for $y$
	cudaMalloc((void**) &y_d,
			sizeof(float) * (storedSizeX / 2) * (NUM_PATCHES_X + 1)
					* (storedSizeX / 2) * (NUM_PATCHES_Y + 1));
/// Space for $y_k$
	cudaMalloc((void**) &y_k_d,
			sizeof(float) * (storedSizeX / 2) * (NUM_PATCHES_X + 1)
					* (storedSizeX / 2) * (NUM_PATCHES_Y + 1));
	//(storedSizeX / 2) * (NUM_PATCHES_X +1) * 	(storedSizeX / 2) * (NUM_PATCHES_Y +1)
/// Space for v_1 (temporary complex patches)
	cudaMalloc((void**) &v_tmp_cmplx_d,
			sizeof(cufftComplex) * FFT_SIZE * (FFT_SIZE / 2 + 1) * NUM_PATCHES_X
					* NUM_PATCHES_Y);
// Just one single float pointer
	cudaMalloc((void**) &f_n_d, sizeof(float));
// Just one single unsigned int pointer
	cudaMalloc((void**) &count_d, sizeof(unsigned int));
// The buffer for adding the sum of all thread blocks together
	cudaMalloc((void**) &f_n_part_sums_d,
			sizeof(float)
					* roundBlkSizeUp(((storedSizeX / 2) * (NUM_PATCHES_X +1) * (storedSizeX / 2) * (NUM_PATCHES_Y +1)),1024));
// The helper-bundle-thingy-struct
	cudaMalloc((void**) &helper_struct_d, sizeof(store_f_X_T_1_informations));
/// Space for adding $||}nabla f||^2$ abs sums together
	cudaMalloc((void**) &(helper_struct_h->abs_vec_nabla_f_part_sums),
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
/// Space for adding $||\delta f||^2$ abs sums together
	cudaMalloc((void**) &(helper_struct_h->abs_vec_delta_f_part_sums),
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
/// Space for adding $<\nabla f_o,\ x_o - f>$ sums together
	cudaMalloc(
			(void**) &(helper_struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d),
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
/// Space for adding $<\nabla f,\ \delta f>$ sums together
	cudaMalloc(
			(void**) &(helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums),
			sizeof(float) * 32768);
	streamCallback->finished = false;
	streamCallback->helper_struct_d = helper_struct_d;
	streamCallback->helper_struct_h = helper_struct_h;
	streamCallback->part_sums_var_h = part_sums_var_h;
	streamCallback->delta_nabla_f_part_sums_h = delta_nabla_f_part_sums_h;
	streamCallback->f_n_h = f_n_h;
	cudaGetDevice(&dev);
	cudaMemsetAsync(f_n_d, 0, sizeof(float), stream);
// get f over to the GPU
	cudaMemcpyAsync(f_d, f_h,
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y,
			cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(y_k_d, y_k_h,
			sizeof(float) * (storedSizeX / 2) * (NUM_PATCHES_X + 1)
					* (storedSizeX / 2) * (NUM_PATCHES_Y + 1),
			cudaMemcpyHostToDevice, stream);
/// set some values for $\alpha$ and $\beta§
	helper_struct_h->alpha = 0.5;
	helper_struct_h->beta = 0.5;
// get the helper struct over
	cudaMemcpyAsync(helper_struct_d, helper_struct_h,
			sizeof(store_f_X_T_1_informations), cudaMemcpyHostToDevice, stream);
	cudaDeviceSynchronize();
	snprintf(str, 80, "pre-loop-x_d_X.pgm");
		str[79] = '\0';
		output_size_X(x_d, str);
//puts("Name:\n");
	cudaError_t error = cudaGetLastError();
//puts(cudaGetErrorName(error));
//puts("\nDescription:\n");
//puts(cudaGetErrorString(error));
#if 1
	cufftHandle plan_x_p_F;
	{
		cufftCreate(&plan_x_p_F);
		int inembed[] = { 1, MASK_X_F_LOAD };
		int onembed[] = { 1, FFT_SIZE / 2 + 1 };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_x_p_F, 2, n, inembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, onembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), CUFFT_R2C,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_x_p_F, stream);
		cufftXtSetCallback(plan_x_p_F, ((void**) &_h_load_x_p_F),
				CUFFT_CB_LD_REAL, NULL);
	}
	/**
	 * $X_1$
	 */
	cufftHandle plan_f_X_1_l_F;
	{
		int inembed[] = { 1, MASK_X_F_LOAD };
		int onembed[] = { 1, FFT_SIZE / 2 + 1 };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_1_l_F, 2, n, inembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, onembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), CUFFT_R2C,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_1_l_F, stream);
		cufftXtSetCallback(plan_f_X_1_l_F, ((void**) &_h_load_f_X_1_F),
				CUFFT_CB_LD_REAL, NULL);
		cufftXtSetCallback(plan_f_X_1_l_F, ((void**) &_h_store_f_X_fft_m_x_F),
				CUFFT_CB_ST_COMPLEX, ((void**) &x_p_d));
	}
	cufftHandle plan_f_X_1_s_F;
	{
		cufftCreate(&plan_f_X_1_s_F);
		int inembed[] = { 1, FFT_SIZE / 2 + 1 };
		int onembed[] = { 1, MASK_X_F_LOAD };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_1_s_F, 2, n, inembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), onembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, CUFFT_C2R,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_1_s_F, stream);
		cufftXtSetCallback(plan_f_X_1_s_F, ((void**) &_h_store_f_X_y_p_v_1_F),
				CUFFT_CB_ST_REAL, NULL);
	}
	/**
	 * $X^T_1$ and $X^T_2$
	 */
	cufftHandle plan_f_X_T_l_F;
	{
		cufftCreate(&plan_f_X_T_l_F);
		int inembed[] = { 1, MASK_X_F_LOAD };
		int onembed[] = { 1, FFT_SIZE / 2 + 1 };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_T_l_F, 2, n, inembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, onembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), CUFFT_R2C,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_T_l_F, stream);
		cufftXtSetCallback(plan_f_X_T_l_F, ((void**) &_h_load_v_3_X_T_F),
				CUFFT_CB_LD_REAL, NULL);
		cufftXtSetCallback(plan_f_X_T_l_F, ((void**) &_h_store_f_X_T_fft_m_x_F),
				CUFFT_CB_ST_COMPLEX, ((void**) &x_p_d));
	}
	/**
	 * $X^T_1$
	 */
	cufftHandle plan_f_X_T_1_nabla_tilde_f_even_b_F;
	{
		cufftCreate(&plan_f_X_T_1_nabla_tilde_f_even_b_F);
		int inembed[] = { 1, FFT_SIZE / 2 + 1 };
		int onembed[] = { 1, MASK_X_F_LOAD };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_T_1_nabla_tilde_f_even_b_F, 2, n, inembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), onembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, CUFFT_C2R,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_T_1_nabla_tilde_f_even_b_F, stream);
		cufftXtSetCallback(plan_f_X_T_1_nabla_tilde_f_even_b_F,
				((void**) &_h_store_f_X_T_1_nabla_tilde_f_even_b_F),
				CUFFT_CB_ST_REAL, ((void**) &helper_struct_d));
		cufftXtSetCallbackSharedSize(plan_f_X_T_1_nabla_tilde_f_even_b_F,
				CUFFT_CB_ST_REAL, sizeof(float) * 32);
	}
	cufftHandle plan_f_X_T_1_nabla_tilde_f_uneven_b_F;
	{
		cufftCreate(&plan_f_X_T_1_nabla_tilde_f_uneven_b_F);
		int inembed[] = { 1, FFT_SIZE / 2 + 1 };
		int onembed[] = { 1, MASK_X_F_LOAD };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_T_1_nabla_tilde_f_uneven_b_F, 2, n, inembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), onembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, CUFFT_C2R,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_T_1_nabla_tilde_f_uneven_b_F, stream);
		cufftXtSetCallback(plan_f_X_T_1_nabla_tilde_f_uneven_b_F,
				((void**) &_h_store_f_X_T_1_nabla_tilde_f_uneven_b_F),
				CUFFT_CB_ST_REAL, ((void**) &helper_struct_d));
		cufftXtSetCallbackSharedSize(plan_f_X_T_1_nabla_tilde_f_uneven_b_F,
				CUFFT_CB_ST_REAL, sizeof(float) * 32);
	}
	/**
	 * $X^T_2$
	 */
	cufftHandle plan_f_X_T_2_delta_tilde_f_even_b_F;
	{
		cufftCreate(&plan_f_X_T_2_delta_tilde_f_even_b_F);
		int inembed[] = { 1, FFT_SIZE / 2 + 1 };
		int onembed[] = { 1, MASK_X_F_LOAD };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_T_2_delta_tilde_f_even_b_F, 2, n, inembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), onembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, CUFFT_C2R,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_T_2_delta_tilde_f_even_b_F, stream);
		cufftXtSetCallback(plan_f_X_T_2_delta_tilde_f_even_b_F,
				((void**) &_h_store_f_X_T_2_delta_tilde_f_even_b_F),
				CUFFT_CB_ST_REAL, ((void**) &helper_struct_d));
		cufftXtSetCallbackSharedSize(plan_f_X_T_2_delta_tilde_f_even_b_F,
				CUFFT_CB_ST_REAL, sizeof(float) * 32);
	}
	cufftHandle plan_f_X_T_2_delta_tilde_f_uneven_b_F;
	{
		cufftCreate(&plan_f_X_T_2_delta_tilde_f_uneven_b_F);
		int inembed[] = { 1, FFT_SIZE / 2 + 1 };
		int onembed[] = { 1, MASK_X_F_LOAD };
		int n[] = { FFT_SIZE, FFT_SIZE };
		cufftPlanMany(&plan_f_X_T_2_delta_tilde_f_uneven_b_F, 2, n, inembed, 1,
				FFT_SIZE * (FFT_SIZE / 2 + 1), onembed, 1,
				MASK_X_F_LOAD * MASK_X_F_LOAD, CUFFT_C2R,
				NUM_PATCHES_X * NUM_PATCHES_Y);
		cufftSetStream(plan_f_X_T_2_delta_tilde_f_uneven_b_F, stream);
		cufftXtSetCallback(plan_f_X_T_2_delta_tilde_f_uneven_b_F,
				((void**) &_h_store_f_X_T_2_delta_tilde_f_uneven_b_F),
				CUFFT_CB_ST_REAL, ((void**) &helper_struct_d));
		cufftXtSetCallbackSharedSize(plan_f_X_T_2_delta_tilde_f_uneven_b_F,
				CUFFT_CB_ST_REAL, sizeof(float) * 32);
	}
#endif
	cufftExecR2C(plan_x_p_F, x_d, x_p_d);
// Zero them out!
	setFloatDeviceZero(v_3_d,
			((storedSizeX / 2) * (NUM_PATCHES_X + 1) * (storedSizeX / 2)
					* (NUM_PATCHES_Y + 1)), 128, stream);
//printf("before main loop");
	int ksksk = 0;
	snprintf(str, 80, "pre-loop-y_k_d_X.pgm");
	str[79] = '\0';
	output_size_X(y_k_d, str);
	do {
		for (int b = 0; b < 5/*M*/; b++) {
			///$X_1$
			printf("b=%d ", b);
			snprintf(str, 80, "pre-anything-f-iter#%d_F.pgm", b);
			str[79] = '\0';
			output_size_F(f_d, str);
			cufftExecR2C(plan_f_X_1_l_F, f_d, v_tmp_cmplx_d);
			CUDA_ERROR_CHECK
			setFloatDeviceZero(y_d,
					((storedSizeX / 2) * (NUM_PATCHES_X + 1) * (storedSizeX / 2)
							* (NUM_PATCHES_Y + 1)), 128, stream);
			cufftExecC2R(plan_f_X_1_s_F, v_tmp_cmplx_d, y_d);
			CUDA_ERROR_CHECK
			snprintf(str, 80, "post-first-block-y-iter#%d_X.pgm", b);
			str[79] = '\0';
			output_size_X(f_d, str);
			///$v_3 \gets y - y'_k$
			kernel_v_3_gets_y_min_y_k_and_f_n_gets_abs_bracketo_y_min_y_i_bracketc_sqr<<<
					roundBlkSizeUp(((storedSizeX / 2) * (NUM_PATCHES_X +1) * (storedSizeX / 2) * (NUM_PATCHES_Y +1)),1024),
					1024, 0, stream>>>(v_3_d, y_d, y_k_d, f_n_part_sums_d,
					f_n_d, count_d);
			CUDA_ERROR_CHECK
				//printf("Past X_1");
				/// $X^T_1$
			cufftExecR2C(plan_f_X_T_l_F, v_3_d, v_tmp_cmplx_d);
			CUDA_ERROR_CHECK
				//printf("In X^T_1");
			if (b % 2 == 0) { // b is even
				cufftExecC2R(plan_f_X_T_1_nabla_tilde_f_even_b_F, v_tmp_cmplx_d,
						v_3_d);
			} else { // b is uneven
				cufftExecC2R(plan_f_X_T_1_nabla_tilde_f_uneven_b_F,
						v_tmp_cmplx_d, v_3_d);
			}
			CUDA_ERROR_CHECK
				///$X_2$
			cufftExecR2C(plan_f_X_1_l_F, helper_struct_h->vec_nabla_f_o,
					v_tmp_cmplx_d);
			CUDA_ERROR_CHECK
			setFloatDeviceZero(y_d,
					((storedSizeX / 2) * (NUM_PATCHES_X + 1) * (storedSizeX / 2)
							* (NUM_PATCHES_Y + 1)), 128, stream);
			CUDA_ERROR_CHECK
			cufftExecC2R(plan_f_X_1_s_F, v_tmp_cmplx_d, v_3_d);
			CUDA_ERROR_CHECK
				///$X^T_2$
			cufftExecR2C(plan_f_X_T_l_F, v_3_d, v_tmp_cmplx_d);
			CUDA_ERROR_CHECK
			if (b % 2 == 0) { // b is even
				cufftExecC2R(plan_f_X_T_2_delta_tilde_f_even_b_F, v_tmp_cmplx_d,
						v_3_d);
			} else { // b is uneven
				cufftExecC2R(plan_f_X_T_2_delta_tilde_f_uneven_b_F,
						v_tmp_cmplx_d, v_3_d);
			}
			CUDA_ERROR_CHECK
			cudaMemcpyAsync((void*) helper_struct_h, (void*) helper_struct_d,
					sizeof(store_f_X_T_1_informations), cudaMemcpyDeviceToHost,
					stream);
			CUDA_ERROR_CHECK
			if (b == 0) { // first iteration
				cudaStreamSynchronize(stream);
				CUDA_ERROR_CHECK
					/// Space for adding $||\delta\nabla f||^2$ on host
				cudaMallocHost(&delta_nabla_f_part_sums_h,
						sizeof(float) * helper_struct_h->block_num, 0);
				CUDA_ERROR_CHECK
				cudaMallocHost(&part_sums_var_h,
						sizeof(float) * helper_struct_h->block_num, 0);
				CUDA_ERROR_CHECK
				cudaMallocHost(
						(void**) &(streamCallback->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h),
						sizeof(float) * helper_struct_h->block_num);
				CUDA_ERROR_CHECK
				streamCallback->delta_nabla_f_part_sums_h =
						delta_nabla_f_part_sums_h;
				streamCallback->part_sums_var_h = part_sums_var_h;
			}
			cudaMemcpyAsync((void*) delta_nabla_f_part_sums_h,
					(void*) helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums,
					sizeof(float) * helper_struct_h->block_num,
					cudaMemcpyDeviceToHost, stream);
			CUDA_ERROR_CHECK
			if (b % 2 == 0) // even
				cudaMemcpyAsync((void*) part_sums_var_h,
						(void*) helper_struct_h->abs_vec_nabla_f_part_sums,
						sizeof(float) * helper_struct_h->block_num,
						cudaMemcpyDeviceToHost, stream);
			else
				// uneven
				cudaMemcpyAsync((void*) part_sums_var_h,
						(void*) helper_struct_h->abs_vec_delta_f_part_sums,
						sizeof(float) * helper_struct_h->block_num,
						cudaMemcpyDeviceToHost, stream);
			CUDA_ERROR_CHECK
			cudaMemcpyAsync((void*) f_n_h, (void*) f_n_d, sizeof(float),
					cudaMemcpyDeviceToHost, stream);
			CUDA_ERROR_CHECK
			cudaMemcpyAsync(
					(void*) streamCallback->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_h,
					(void*) helper_struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d,
					sizeof(float) * helper_struct_h->block_num,
					cudaMemcpyDeviceToHost, stream);
			CUDA_ERROR_CHECK
			cudaStreamAddCallback(stream,
					(cudaStreamCallback_t) optimizeFcallback,
					(void*) streamCallback, 0);
			CUDA_ERROR_CHECK
			if (b % 2 == 0) { // even, we have to check if we are finished
				cudaStreamSynchronize(stream);
//				if (streamCallback->finished)
//					// we are finished with this optimization
//					goto end_loop;
			}
			CUDA_ERROR_CHECK
			cudaMemcpyAsync(helper_struct_d, helper_struct_h,
					sizeof(store_f_X_T_1_informations), cudaMemcpyHostToDevice,
					stream);
			CUDA_ERROR_CHECK
		}
	} while (ksksk++ < 1);
	end_loop: cudaMemcpyAsync((void*) f_h, (void*) f_d,
			sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y,
			cudaMemcpyDeviceToHost, stream);
	CUDA_ERROR_CHECK
	cudaStreamSynchronize(stream);
	CUDA_ERROR_CHECK
	cudaFree(f_d);
	CUDA_ERROR_CHECK
	cudaFree((helper_struct_h->vec_f_o));
	CUDA_ERROR_CHECK
	cudaFree((helper_struct_h->vec_nabla_f_o));
	CUDA_ERROR_CHECK
	cudaFree(x_p_d);
	CUDA_ERROR_CHECK
	cudaFree(x_d);
	CUDA_ERROR_CHECK
	cudaFree(v_3_d);
	CUDA_ERROR_CHECK
	cudaFree(y_d);
	CUDA_ERROR_CHECK
	cudaFree(y_k_d);
	CUDA_ERROR_CHECK
	cudaFree(v_tmp_cmplx_d);
	CUDA_ERROR_CHECK
	cudaFree(f_n_d);
	CUDA_ERROR_CHECK
	cudaFree(count_d);
	CUDA_ERROR_CHECK
	cudaFree(f_n_part_sums_d);
	CUDA_ERROR_CHECK
	cudaFree(helper_struct_d);
	CUDA_ERROR_CHECK
	cudaFree((helper_struct_h->abs_vec_nabla_f_part_sums));
	CUDA_ERROR_CHECK
	cudaFree((helper_struct_h->abs_vec_delta_f_part_sums));
	CUDA_ERROR_CHECK
	cudaFree(
			(helper_struct_h->nabla_f_o_scalar_prod_bracketo_x_o_minus_new_f_bracketc_part_sums_d));
	CUDA_ERROR_CHECK
	cudaFree((helper_struct_h->nabla_f_scalar_prod_delta_f_part_sums));
	CUDA_ERROR_CHECK

	free(helper_struct_h);
	CUDA_ERROR_CHECK
	free(streamCallback);
	CUDA_ERROR_CHECK

	cudaFreeHost(f_n_h);
	CUDA_ERROR_CHECK
	cudaFreeHost(delta_nabla_f_part_sums_h);
	CUDA_ERROR_CHECK
	cudaFreeHost(part_sums_var_h);
	CUDA_ERROR_CHECK

//TODO: cleanup of all the allocated resources
	return 0;
}

void gen_spike_f() {
	//spiked_f_h = (float*) malloc(	sizeof(float) * NUM_PATCHES_X * NUM_PATCHES_Y * SIZE_F * SIZE_F);
	for (int patch = 0; patch < (NUM_PATCHES_X * NUM_PATCHES_Y); patch++)
		for (int i = 0; i < (SIZE_F * SIZE_F); i++)
			spiked_f_h[patch * SIZE_F * SIZE_F + i] = (1 / (SIZE_F * SIZE_F));
	//(SIZE_F * SIZE_HALF_F + SIZE_HALF_F) == i ? 1 : 0;
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	void *d = NULL;
	float* f_h[NUM_TEST_PARALLEL_RUNS];
	gen_spike_f();
	{
		int cnt = 0;
		for (int i = 0; i < (sizeof(spiked_f_h) / sizeof(spiked_f_h[0])); i++)
			if (spiked_f_h[i] != 0)
				cnt++;
		float** spikedAdresses = (float**) malloc(sizeof(float*) * cnt);
		int* spikedPositions = (int*) malloc(sizeof(int) * cnt);
		cnt = 0;
		for (int i = 0; i < (sizeof(spiked_f_h) / sizeof(spiked_f_h[0])); i++)
			if (spiked_f_h[i] != 0) {
				spikedAdresses[cnt] = &spiked_f_h[i];
				spikedPositions[cnt++] = i;
			}
	}
	float* image_folded_conv;
	float* image_unfolded_conv;
	cudaMallocHost((void**) (void*) &image_folded_conv,
			512 * 512 * sizeof(float));
	cudaMallocHost((void**) (void*) &image_unfolded_conv,
			512 * 512 * sizeof(float));
	getCallbacks();
	cudaStream_t stream[NUM_TEST_PARALLEL_RUNS];
	for (int j = 0; j < NUM_TEST_PARALLEL_RUNS; j++) {
		f_h[j] = NULL;
		stream[j] = NULL;
		cudaMallocHost((void**) (void*) &(f_h[j]),
				sizeof(float) * (SIZE_F * SIZE_F) * NUM_PATCHES_X
						* NUM_PATCHES_Y, 0);
		for (int i = 0; i < (SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y;
				i++)
			(f_h[j])[i] = 0;
		for (int i = 0; i < (512 * 512); i++) {
			image_folded_conv[i] = image_folded.pixel_data[i * 3] / 255.;
			image_unfolded_conv[i] = image_unfolded.pixel_data[i * 3] / 255.;
		}
		memcpy(f_h[j], spiked_f_h,
				(SIZE_F * SIZE_F) * NUM_PATCHES_X * NUM_PATCHES_Y);
		cudaStreamCreate(&(stream[j]));
		optimizeF(f_h[j], image_folded_conv, image_unfolded_conv, stream[j]);
		printf("Finished round #%i of %i\n", j + 1, NUM_TEST_PARALLEL_RUNS);
	}
	cudaDeviceSynchronize();
	CUDA_ERROR_CHECK

// Open the PGM output file
	FILE* file = fopen("output.pgm", "wb");

// Write the file header
	fprintf(file, "P5\n"
			"%d %d\n"
			"255\n", (SIZE_F * NUM_PATCHES_Y), (SIZE_F * NUM_PATCHES_X));

// Write the content: remember, data is saved in x-major order, both the patches themself and the data inside of the patches
	double max = 0;
	for (int x = 0; x < NUM_PATCHES_X * SIZE_F; x++)
		for (int y = 0; y < NUM_PATCHES_Y * SIZE_F; y++)
			max = (f_h[0])[x * (NUM_PATCHES_Y * SIZE_F) + y] > max ?
					(f_h[0])[x * (NUM_PATCHES_Y * SIZE_F) + y] : max;
	// Write the content: remember, data is saved in x-major order, both the patches themself and the data inside of the patches
	for (int xPatch = 0; xPatch < (NUM_PATCHES_X); xPatch++)
		for (int xPos = 0; xPos < SIZE_F; xPos++)
			for (int yPatch = 0; yPatch < (NUM_PATCHES_Y); yPatch++)
				for (int yPos = 0; yPos < SIZE_F; yPos++)
					fputc(
							((char) ((uint) ((f_h[0])[(xPatch * NUM_PATCHES_Y
									+ yPatch) * SIZE_F * SIZE_F + xPos * SIZE_F
									+ yPos] * (255 / max)))), file);
	fclose(file);
	/*{
	 // Open the PGM output file
	 FILE* file = fopen("output_unmangeled_input.pgm", "wb");

	 // Write the file header
	 fprintf(file, "P5\n"
	 "%d %d\n"
	 "255\n", 512, 512);
	 for(int i=0; i<512*512;i++){
	 fputc(image_unfolded.pixel_data[i*3], file);
	 }
	 fclose(file);

	 }*/
	puts("Finished Writing.");
	printf("max=%f", (float) max);
	cudaFreeHost(image_folded_conv);
	CUDA_ERROR_CHECK
	cudaFreeHost(image_unfolded_conv);
	CUDA_ERROR_CHECK
	for (int j = 0; j < NUM_TEST_PARALLEL_RUNS; j++)
		cudaFreeHost(f_h[j]);
	CUDA_ERROR_CHECK
//	CUDA_CHECK_RETURN(cudaMalloc((void** ) &d, sizeof(int) * WORK_SIZE));
//	CUDA_CHECK_RETURN(
//			cudaMemcpy(d, idata, sizeof(int) * WORK_SIZE,
//					cudaMemcpyHostToDevice));
//	cufftHandle plan;
//	cufftCreate(&plan);
////cufftType b =  R2C;
//	cufftComplex odata[MASK_X_F_LOAD][MASK_X_F_LOAD / 2 + 1];
//	int inembed[] = { 1, MASK_X_F_LOAD };
//	int onembed[] = { 1, FFT_SIZE / 2 + 1 };
//	int n[] = { FFT_SIZE, FFT_SIZE };
//	cufftPlanMany(&plan, 2, n, inembed, 1, MASK_X_F_LOAD * MASK_X_F_LOAD, onembed, 1,
//			FFT_SIZE * (FFT_SIZE / 2 + 1), CUFFT_R2C,
//			NUM_PATCHES_X * NUM_PATCHES_Y);
//	CUDA_CHECK_RETURN(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
//	CUDA_CHECK_RETURN(cudaGetLastError());
//	CUDA_CHECK_RETURN(
//			cudaMemcpy(odata, d, sizeof(int) * WORK_SIZE,
//					cudaMemcpyDeviceToHost));
//	CUDA_CHECK_RETURN(cudaFree((void* ) d));
	CUDA_CHECK_RETURN(cudaDeviceReset());
	puts(cudaGetErrorString(cudaGetLastError()));
	return cudaGetLastError();
}
