//=========================================================================================
//							DETAILS ABOUT THE SUBMISSION
//=========================================================================================
// Name: Jokubas Liutkus
// ID: 1601768
//
// Goals Achieved (all goals were achieved):
//	1. block scan +
//	2. full scan for large vectors +
//	3. Bank conflict avoidance optimization (BCAO) +
//
// Your time, in milliseconds to execute the different scans on a vector of 10,000,000 entries:
//	∗ Block scan without BCAO:
//	∗ Block scan with BCAO:
//	∗ Full scan without BCAO:
//	∗ Full scan with BCAO:
//
// CPU model: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz x 4 (lab machine)
// GPU model: GeForce GTX 960 (lab machine)
//
// Improvements/Additional features:
//	1. All the multiplications were replaced with bit shifting which improved the performance
//	2. Extract sum phase was merged with the main block scan function removing additional kernel call
//	3. Padded the shared memory array of the last block to zero while loading in the data from device memory
//	   to get the faster operations on the last block elements.
//	4. Saving the last element of each block to the local variable from the shared memory
//	   before running reduction and distribution phases rather than loading it later from a global memory
//
//=========================================================================================
//=========================================================================================
//=========================================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}

// Block size and its double version
#define BLOCK_SIZE 512
#define BLOCK_SIZE_TWICE BLOCK_SIZE*2

// for avoiding bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
		((n) >> NUM_BANKS + (n) >> (LOG_NUM_BANKS << 1))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif


static void compare_results(const int *vector1, const int *vector2,
		int numElements) {
	for (int i = 0; i < numElements; ++i) {
//		printf("%d ----------- %d\n", vector1[i], vector2[i]);
		if (fabs(vector1[i] - vector2[i]) > 1e-5f) {
			printf("%d ----------- %d\n", vector1[i], vector2[i]);
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}

__host__
void sequential_scan(int *g_idata, int *g_odata, int n) {

	g_odata[0] = 0;
	for (int i = 1; i < n; i++) {
		g_odata[i] = g_odata[i - 1] + g_idata[i - 1];
	}
}

__global__
void add_to_block(int *block, int len_block, int *SUM) {
	int s = SUM[blockIdx.x];

	int addr = blockIdx.x * (blockDim.x << 1) + threadIdx.x;

	__syncthreads();
	if (addr < len_block)
		block[addr] += s;
	if (addr + blockDim.x < len_block)
		block[addr + blockDim.x] += s; // TODO need to check properly

}

__global__
void block_scan_full_BCAO(int *g_idata, int *g_odata, int n, int *SUM,
		int add_last) {

	const int blockSize = (BLOCK_SIZE << 1);

	__shared__ int temp[blockSize + (BLOCK_SIZE / 8)]; // allocated on invocation

	int thid = threadIdx.x;
	int thid_shift = thid << 1;
	int blockId = blockIdx.x * (BLOCK_SIZE << 1);
	int offset = 0;  //1;
	int last = 0;

	int ai = thid;
	int bi = thid + BLOCK_SIZE;  //(n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if (blockId + ai < n) {
		temp[ai + bankOffsetA] = g_idata[blockId + ai];
	} else
		temp[ai + bankOffsetA] = 0;
	if (blockId + bi < n) {
		temp[bi + bankOffsetB] = g_idata[blockId + bi];
	} else
		temp[bi + bankOffsetB] = 0;

	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		last =
				temp[blockSize - 1 + CONFLICT_FREE_OFFSET((BLOCK_SIZE << 1) - 1)];

	for (int d = BLOCK_SIZE ; d > 0; d >>= 1) // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = ((thid_shift + 1) << offset) - 1;
			int bi = ((thid_shift + 2) << offset) - 1;

			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}

		offset++;
	}

	if (thid == 0) {
		temp[blockSize - 1 + CONFLICT_FREE_OFFSET(blockSize - 1)] = 0;
	}

	for (int d = 1; d < blockSize; d <<= 1) // traverse down tree & build scan
			{
		offset--;
		__syncthreads();
		if (thid < d) {
			int ai = ((thid_shift + 1) << offset) - 1;
			int bi = ((thid_shift + 2) << offset) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		SUM[blockIdx.x] = temp[blockSize - 1
				+ CONFLICT_FREE_OFFSET(blockSize - 1)] + last;
	if (blockId + ai < n)
		g_odata[blockId + ai] = temp[ai + bankOffsetA];
	if (blockId + bi < n)
		g_odata[blockId + bi] = temp[bi + bankOffsetB];
}

__host__
void full_block_scan_BCAO(int *h_IN, int *h_OUT, int len) {

	// -----------------------------------------------------------
	// 							INITIALIZATION
	// -----------------------------------------------------------

	// error code to check return values for CUDA class
	cudaError_t err = cudaSuccess;

	// size to allocate for the vectors
	size_t size = len * sizeof(int);

	// create device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	// allocate memory for all the possible vectors needed for the execution
	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, size);
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, size);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1 = NULL;
	err = cudaMalloc((void**) &d_SUM_1,
			(1 + ((len - 1) / (BLOCK_SIZE * 2))) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_1_Scanned,
			(1 + ((len - 1) / (BLOCK_SIZE * 2))) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2 = NULL;
	err = cudaMalloc((void**) &d_SUM_2, (BLOCK_SIZE << 1) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_2_Scanned,
			(BLOCK_SIZE << 1) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	// copy the memory from the host to the device
	err = cudaMemcpy(d_IN, h_IN, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");

	// size of the grid for each level
	int blocksPerGridLevel1 = 1 + ((len - 1) / (BLOCK_SIZE * 2));
	int blocksPerGridLevel2 = 1 + ceil(blocksPerGridLevel1 / (BLOCK_SIZE << 1));
	int blocksPerGridLevel3 = 1 + ceil(blocksPerGridLevel2 / (BLOCK_SIZE << 1));

	// -----------------------------------------------------------
	// 							EXECUTION
	// -----------------------------------------------------------

	// choosing the level on which to run the kernels
	// based on the size of the grids

	// if level one grid size is equal to 1 then single
	// LEVEL 1 is enough to scan the whole array
	if (blocksPerGridLevel1 == 1) {
		// record the start time
		cudaEventRecord(d_start, 0);
		printf("lanuched1!\n");
		// execute the actual kernel
		block_scan_full_BCAO<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT,
				len,
				NULL, 0);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// if level two grid size is equal to 1 then two (LEVEL 2)
	// scans are required to scan the whole array
	else if (blocksPerGridLevel2 == 1) {
		// record the start time
		cudaEventRecord(d_start, 0);
		printf("lanuched2!\n");
		// execute the actual kernels
		printf("%d\n", blocksPerGridLevel1);
		block_scan_full_BCAO<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT,
				len, d_SUM_1, 1);
		block_scan_full_BCAO<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
				d_SUM_1_Scanned, blocksPerGridLevel1, NULL, 0);
		add_to_block<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_OUT, len,
				d_SUM_1_Scanned);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// if level 3 grid size is equal to 1 then three (LEVEL 3)
	// scans are required to scan the whole array
	else if (blocksPerGridLevel3 == 1) {
		// record the start time
		cudaEventRecord(d_start, 0);
		printf("lanuched3!\n");
		// execute the actual kernels
		block_scan_full_BCAO<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT,
				len, d_SUM_1, 1);
		block_scan_full_BCAO<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
				d_SUM_1_Scanned, blocksPerGridLevel1, d_SUM_2, 1);
		block_scan_full_BCAO<<<1, BLOCK_SIZE>>>(d_SUM_2, d_SUM_2_Scanned,
				blocksPerGridLevel2, NULL, 0);
		add_to_block<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1_Scanned,
				blocksPerGridLevel1, d_SUM_2_Scanned);
		add_to_block<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_OUT, len,
				d_SUM_1_Scanned);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// if none of the conditions above is met, it means that the array is too
	// large to be scanned in 3 level scan, therefore we print the error message
	// and return
	else {
		fprintf(stderr,
				"The array size=%d is too large to be scanned with level 3 scan!\n",
				len);

		// using goto is discouraged, however, in such situations
		// where in the error conditions exit or cleanup is required
		// it is considered idiomatic
		goto cleanup;
	}

	// check whether the run was successful
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	// get the duration it took for the kernels to execute
	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	// print the time elapsed
	printf(
			"Full block with bank avoidance scan with %d elements took = %.f5mSecs\n",
			len, d_msecs);

	// copy the result from the device back to the host
	err = cudaMemcpy(h_OUT, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array OUT from device to host");

	// -----------------------------------------------------------
	// 							CLEANUP
	// -----------------------------------------------------------

	cleanup:
	// Free device global memory
	CUDA_ERROR(cudaFree(d_IN), "Failed to free device vector IN");
	CUDA_ERROR(cudaFree(d_OUT), "Failed to free device vector OUT");
	CUDA_ERROR(cudaFree(d_SUM_1), "Failed to free device vector SUM_1");
	CUDA_ERROR(cudaFree(d_SUM_1_Scanned),"Failed to free device vector SUM_1_Scanned");
	CUDA_ERROR(cudaFree(d_SUM_2), "Failed to free device vector SUM_2");
	CUDA_ERROR(cudaFree(d_SUM_2_Scanned), "Failed to free device vector SUM_2_Scanned");

	// Clean up the Device timer event objects
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

}

__global__
void block_scan_full(int *g_idata, int *g_odata, int n, int *SUM,
		int add_last) {
	__shared__ int temp[BLOCK_SIZE << 1];  // allocated on invocation

	int thid = threadIdx.x;
	int blockId = blockDim.x * blockIdx.x << 1;
	int offset = 0;
	int last = 0;

	if (blockId + (thid << 1) < n)
		temp[thid << 1] = g_idata[blockId + (thid << 1)]; // load input into shared memory
	if (blockId + (thid << 1) + 1 < n)
		temp[(thid << 1) + 1] = g_idata[blockId + (thid << 1) + 1];

	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		last = temp[(thid << 1) + 1];

	for (int d = BLOCK_SIZE /*n >> 1*/; d > 0; d >>= 1) // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = (((thid << 1) + 1) << offset) - 1;
			int bi = (((thid << 1) + 2) << offset) - 1;
			temp[bi] += temp[ai];
		}
		offset++;
	}

	if (thid == 0) {
		temp[(BLOCK_SIZE << 1) - 1] = 0;
	} // clear the last element

	for (int d = 1; d < (BLOCK_SIZE << 1); d <<= 1) // traverse down tree & build scan
			{
		offset--;
		__syncthreads();
		if (thid < d) {
			int ai = (((thid << 1) + 1) << offset) - 1;
			int bi = (((thid << 1) + 2) << offset) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		SUM[blockIdx.x] = temp[(thid << 1) + 1] + last;
	if (blockId + (thid << 1) < n)
		g_odata[blockId + (thid << 1)] = temp[thid << 1]; // write results to device memory
	if (blockId + (thid << 1) + 1 < n)
		g_odata[blockId + (thid << 1) + 1] = temp[(thid << 1) + 1];
}

__host__
void full_block_scan(int *h_IN, int *h_OUT, int len) {

	// -----------------------------------------------------------
	// 							INITIALIZATION
	// -----------------------------------------------------------

	// error code to check return values for CUDA class
	cudaError_t err = cudaSuccess;

	// size to allocate for the vectors
	size_t size = len * sizeof(int);

	// create device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	// allocate memory for all the possible vectors needed for the execution
	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, size);
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, size);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1 = NULL;
	err = cudaMalloc((void**) &d_SUM_1,
			(1 + ((len - 1) / (BLOCK_SIZE * 2))) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_1_Scanned,
			(1 + ((len - 1) / (BLOCK_SIZE * 2))) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2 = NULL;
	err = cudaMalloc((void**) &d_SUM_2, (BLOCK_SIZE << 1) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_2_Scanned,
			(BLOCK_SIZE << 1) * sizeof(int));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	// copy the memory from the host to the device
	err = cudaMemcpy(d_IN, h_IN, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");

	// size of the grid for each level
	int blocksPerGridLevel1 = 1 + ((len - 1) / (BLOCK_SIZE * 2));
	int blocksPerGridLevel2 = 1 + ceil(blocksPerGridLevel1 / (BLOCK_SIZE << 1));
	int blocksPerGridLevel3 = 1 + ceil(blocksPerGridLevel2 / (BLOCK_SIZE << 1));

	// -----------------------------------------------------------
	// 							EXECUTION
	// -----------------------------------------------------------

	// choosing the level on which to run the kernels
	// based on the size of the grids

	// if level one grid size is equal to 1 then single
	// LEVEL 1 is enough to scan the whole array
	if (blocksPerGridLevel1 == 1) {
		// record the start time
		cudaEventRecord(d_start, 0);

		// execute the actual kernel
		block_scan_full<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, len,
		NULL, 0);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// if level two grid size is equal to 1 then two (LEVEL 2)
	// scans are required to scan the whole array
	else if (blocksPerGridLevel2 == 1) {
		// record the start time
		cudaEventRecord(d_start, 0);

		// execute the actual kernels
		block_scan_full<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, len,
				d_SUM_1, 1);
		block_scan_full<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
				d_SUM_1_Scanned, blocksPerGridLevel1, NULL, 0);
		add_to_block<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_OUT, len,
				d_SUM_1_Scanned);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// if level 3 grid size is equal to 1 then three (LEVEL 3)
	// scans are required to scan the whole array
	else if (blocksPerGridLevel3 == 1) {
		// record the start time
		cudaEventRecord(d_start, 0);

		// execute the actual kernels
		block_scan_full<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, len,
				d_SUM_1, 1);
		block_scan_full<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
				d_SUM_1_Scanned, blocksPerGridLevel1, d_SUM_2, 1);
		block_scan_full<<<1, BLOCK_SIZE>>>(d_SUM_2, d_SUM_2_Scanned,
				blocksPerGridLevel2, NULL, 0);
		add_to_block<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1_Scanned,
				blocksPerGridLevel1, d_SUM_2_Scanned);
		add_to_block<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_OUT, len,
				d_SUM_1_Scanned);

		// record the stop time
		cudaEventRecord(d_stop, 0);
		cudaEventSynchronize(d_stop);
		cudaDeviceSynchronize();
	}

	// if none of the conditions above is met, it means that the array is too
	// large to be scanned in 3 level scan, therefore we print the error message and return
	else {
		fprintf(stderr,
				"The array size=%d is too large to be scanned with level 3 scan!\n",
				len);

		// using goto is discouraged, however, in such situations
		// where in the error conditions exit or cleanup is required
		// it is considered idiomatic
		goto cleanup;
	}

	// check whether the run was successful
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	// get the duration it took for the kernels to execute
	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	// print the time elapsed
	printf("Full block scan with %d elements took = %.f5mSecs\n", len, d_msecs);

	// copy the result from the device back to the host
	err = cudaMemcpy(h_OUT, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array OUT from device to host");

	// -----------------------------------------------------------
	// 							CLEANUP
	// -----------------------------------------------------------

	cleanup:
	// Free device global memory
	CUDA_ERROR(cudaFree(d_IN), "Failed to free device vector IN");
	CUDA_ERROR(cudaFree(d_OUT), "Failed to free device vector OUT");
	CUDA_ERROR(cudaFree(d_SUM_1), "Failed to free device vector SUM_1");
	CUDA_ERROR(cudaFree(d_SUM_1_Scanned),
			"Failed to free device vector SUM_1_Scanned");
	CUDA_ERROR(cudaFree(d_SUM_2), "Failed to free device vector SUM_2");
	CUDA_ERROR(cudaFree(d_SUM_2_Scanned),
			"Failed to free device vector SUM_2_Scanned");

	// Clean up the Device timer event objects
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");
}

/**
 * Host main routine
 */
int main(void) {

// error code to check return calues for CUDA calss
	cudaError_t err = cudaSuccess;

// create host stopwatch times
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

// create device timer
//	cudaEvent_t d_start, d_stop;
//	float d_msecs;
//	cudaEventCreate(&d_start);
//	cudaEventCreate(&d_stop);

// size of the array to add
	int numElements = 10000000;
	size_t size = numElements * sizeof(int);

// allocate the memory on the host for the arrays
	int *h_IN = (int *) malloc(size);
	int *h_OUT = (int *) malloc(size);
	int *h_OUT_CUDA = (int *) malloc(size);

// verify the host allocations
	if (h_IN == NULL || h_OUT == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

// initialise the host input to 1.0f
	for (int i = 0; i < numElements; i++) {
		h_IN[i] = 1; //rand() % 10;
	}

// sequential scan
	sdkStartTimer(&timer);
	sequential_scan(h_IN, h_OUT, numElements);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("Sequential scan on host of %d elements took = %.f5mSecs\n",
			numElements, h_msecs);

// ACtual algorithm

// allocate memory for the device
//	int *d_IN = NULL;
//	err = cudaMalloc((void **) &d_IN, size);
//	CUDA_ERROR(err, "Failed to allocate device vector IN");
//
//	int *d_OUT = NULL;
//	err = cudaMalloc((void**) &d_OUT, size);
//	CUDA_ERROR(err, "Failed to allocate device vector OUT");

// copy the memory from host to device
//	err = cudaMemcpy(d_IN, h_IN, size, cudaMemcpyHostToDevice);
//	CUDA_ERROR(err, "Failed to copy array IN from host to device");
//	int blocksPerGridLevel1 = 1 + ((numElements - 1) / (BLOCK_SIZE * 2));
//
//	cudaEventRecord(d_start, 0);
//	block_scan_full_BCAO<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT,
//			numElements,
//			NULL, 0);
//
//	cudaEventRecord(d_stop, 0);
//	cudaEventSynchronize(d_stop);
//
//	cudaDeviceSynchronize();
//	err = cudaGetLastError();
//	CUDA_ERROR(err, "Failed to launch block scan kernel");
//
//	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
//	CUDA_ERROR(err, "Failed to get elapsed time");
//
//	printf("Block scan with single thread of %d elements took = %.f5mSecs\n",
//			numElements, d_msecs);
//
//	err = cudaMemcpy(h_OUT_CUDA, d_OUT, size, cudaMemcpyDeviceToHost);
//	CUDA_ERROR(err, "Failed to copy array OUT from device to host");
//
//	//// cleanup
//	// Free device global memory
//	err = cudaFree(d_IN);
//	CUDA_ERROR(err, "Failed to free device vector A");
//	err = cudaFree(d_OUT);
//	CUDA_ERROR(err, "Failed to free device vector B");
//
//	compare_results(h_OUT, h_OUT_CUDA, numElements);
////
	full_block_scan_BCAO(h_IN, h_OUT_CUDA, numElements);

	compare_results(h_OUT, h_OUT_CUDA, numElements);

	full_block_scan(h_IN, h_OUT_CUDA, numElements);

	compare_results(h_OUT, h_OUT_CUDA, numElements);

// Free host memory
	free(h_IN);
	free(h_OUT);
	free(h_OUT_CUDA);

// Clean up the Host timer
	sdkDeleteTimer(&timer);

// Clean up the Device timer event objects
//	cudaEventDestroy(d_start);
//	cudaEventDestroy(d_stop);
//
//// Reset the device and exit
//	err = cudaDeviceReset();
//	CUDA_ERROR(err, "Failed to reset the device");

	return 0;

}
