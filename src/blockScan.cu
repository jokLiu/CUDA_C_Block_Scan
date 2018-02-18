#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
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

#define BLOCK_SIZE 256

// for avoiding bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
		((n) >> NUM_BANKS + (n) >> (LOG_NUM_BANKS * 2))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

// TODO thid variable for shifted version
// TODO fix addition

__global__
void block_scan_full(int *g_idata, int *g_odata, int n, int *SUM, int add_last);

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
void add_to_block(int *block, int *SUM) {
	/*__shared__ */int s;

// TODO check that last block is not used
	s = SUM[blockIdx.x];
	int addr = blockIdx.x * (blockDim.x << 1)
			+ threadIdx.x /*+ (blockDim.x << 1)*/;

	__syncthreads();

	block[addr] += s;
	block[addr + blockDim.x] += s; // TODO need to check properly

}

__global__
void block_scan_full_BCAO(int *g_idata, int *g_odata, int n, int *SUM,
		int add_last) {
	__shared__ int temp[BLOCK_SIZE << 2];  // allocated on invocation

	int thid = threadIdx.x;
	int blockId = blockDim.x * blockIdx.x << 1;
	int offset = 0;  //1;
	int last = 0;

	int ai = thid;
	int bi = thid + BLOCK_SIZE;  //(n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	if (blockId + ai < n)
		temp[ai + bankOffsetA] = g_idata[blockId + ai];
	if (blockId + bi < n)
		temp[bi + bankOffsetB] = g_idata[blockId + bi];

	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		last = temp[(BLOCK_SIZE << 1) - 1
				+ CONFLICT_FREE_OFFSET((BLOCK_SIZE << 1) - 1)];

	for (int d = BLOCK_SIZE /*n >> 1*/; d > 0; d >>= 1) // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = (((thid << 1) + 1) << offset) - 1;
			int bi = (((thid << 1) + 2) << offset) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset++;
	}

	if (thid == 0) {
		temp[(BLOCK_SIZE << 1) - 1 + CONFLICT_FREE_OFFSET((BLOCK_SIZE << 1) - 1)] =
				0;
	}

	for (int d = 1; d < (BLOCK_SIZE << 1); d <<= 1) // traverse down tree & build scan
			{
		offset--;
		__syncthreads();
		if (thid < d) {
			int ai = (((thid << 1) + 1) << offset) - 1;
			int bi = (((thid << 1) + 2) << offset) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		SUM[blockIdx.x] = temp[(BLOCK_SIZE << 1) - 1
				+ CONFLICT_FREE_OFFSET((BLOCK_SIZE << 1) - 1)] + last;

	if (blockId + ai < n)
		g_odata[blockId + ai] = temp[ai + bankOffsetA];
	if (blockId + bi < n)
		g_odata[blockId + bi] = temp[bi + bankOffsetB];
}

__host__
void full_block_scan_BCAO(int *h_IN, int *h_OUT, int len) {
// error code to check return calues for CUDA calss
	cudaError_t err = cudaSuccess;

// create host stopwatch times
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	size_t size = len * sizeof(int);

// create device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, size);
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, size);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1 = NULL;
	err =
			cudaMalloc((void**) &d_SUM_1,
					1
							+ ((len - 1) / (BLOCK_SIZE * 2))/*(int) ceil(len / (BLOCK_SIZE << 1))*/);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	printf("\n%d\n\n", (int) ceil(len / (BLOCK_SIZE << 1)));

	int *d_SUM_1_Scanned = NULL;
	err =
			cudaMalloc((void**) &d_SUM_1_Scanned,
					1
							+ ((len - 1) / (BLOCK_SIZE * 2))/*(int) ceil(len / (BLOCK_SIZE << 1))*/);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2 = NULL;
	err = cudaMalloc((void**) &d_SUM_2, BLOCK_SIZE << 1);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_2_Scanned, BLOCK_SIZE << 1);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

// copy the memory from host to device
	err = cudaMemcpy(d_IN, h_IN, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");

	int blocksPerGridLevel1 = 1 + ((len - 1) / (BLOCK_SIZE * 2));
	int blocksPerGridLevel2 = 1 + ceil(blocksPerGridLevel1 / (BLOCK_SIZE << 1));

	printf("\n%d\n\n", blocksPerGridLevel1);
	printf("\n%d\n\n", blocksPerGridLevel2);

	cudaEventRecord(d_start, 0);
	block_scan_full_BCAO<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, len,
			d_SUM_1, 1);
	block_scan_full_BCAO<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
			d_SUM_1_Scanned, blocksPerGridLevel1, d_SUM_2, 1);
	block_scan_full_BCAO<<<1, BLOCK_SIZE>>>(d_SUM_2, d_SUM_2_Scanned,
			blocksPerGridLevel2, NULL, 0);
	add_to_block<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1_Scanned,
			d_SUM_2_Scanned);
	add_to_block<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_OUT, d_SUM_1_Scanned);
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	printf("Block scan with single thread of %d elements took = %.f5mSecs\n",
			len, d_msecs);

	err = cudaMemcpy(h_OUT, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array OUT from device to host");

//// cleanup
// Free device global memory
	err = cudaFree(d_IN);
	CUDA_ERROR(err, "Failed to free device vector A");
	err = cudaFree(d_OUT);
	CUDA_ERROR(err, "Failed to free device vector B");

// Clean up the Host timer
	sdkDeleteTimer(&timer);

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
//	if (blockId + (thid << 1) < n)
	g_odata[blockId + (thid << 1)] = temp[thid << 1]; // write results to device memory
//	if (blockId + (thid << 1) + 1 < n)
	g_odata[blockId + (thid << 1) + 1] = temp[(thid << 1) + 1];
}

__host__
void full_block_scan(int *h_IN, int *h_OUT, int len) {
	// error code to check return calues for CUDA calss
	cudaError_t err = cudaSuccess;

	// create host stopwatch times
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	size_t size = len * sizeof(int);

	// create device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, size);
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, size);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1 = NULL;
	err =
			cudaMalloc((void**) &d_SUM_1,
					1
							+ ((len - 1) / (BLOCK_SIZE * 2))/*(int) ceil(len / (BLOCK_SIZE << 1))*/);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	printf("\n%d\n\n", (int) ceil(len / (BLOCK_SIZE << 1)));

	int *d_SUM_1_Scanned = NULL;
	err =
			cudaMalloc((void**) &d_SUM_1_Scanned,
					1
							+ ((len - 1) / (BLOCK_SIZE * 2))/*(int) ceil(len / (BLOCK_SIZE << 1))*/);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2 = NULL;
	err = cudaMalloc((void**) &d_SUM_2, BLOCK_SIZE << 1);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_2_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_2_Scanned, BLOCK_SIZE << 1);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	// copy the memory from host to device
	err = cudaMemcpy(d_IN, h_IN, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");

	int blocksPerGridLevel1 = 1 + ((len - 1) / (BLOCK_SIZE * 2));
	int blocksPerGridLevel2 = 1 + ceil(blocksPerGridLevel1 / (BLOCK_SIZE << 1));

	printf("\n%d\n\n", blocksPerGridLevel1);
	printf("\n%d\n\n", blocksPerGridLevel2);

	cudaEventRecord(d_start, 0);
	block_scan_full<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, len,
			d_SUM_1, 1);
	block_scan_full<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
			d_SUM_1_Scanned, blocksPerGridLevel1, d_SUM_2, 1);
	block_scan_full<<<1, BLOCK_SIZE>>>(d_SUM_2, d_SUM_2_Scanned,
			blocksPerGridLevel2, NULL, 0);
	add_to_block<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1_Scanned,
			d_SUM_2_Scanned);
	add_to_block<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_OUT, d_SUM_1_Scanned);
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	printf("Block scan with single thread of %d elements took = %.f5mSecs\n",
			len, d_msecs);

	err = cudaMemcpy(h_OUT, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array OUT from device to host");

	//// cleanup
	// Free device global memory
	err = cudaFree(d_IN);
	CUDA_ERROR(err, "Failed to free device vector A");
	err = cudaFree(d_OUT);
	CUDA_ERROR(err, "Failed to free device vector B");

	// Clean up the Host timer
	sdkDeleteTimer(&timer);

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
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

// size of the array to add
	int numElements = 11111111;
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
	int *d_IN = NULL;
	err = cudaMalloc((void **) &d_IN, size);
	CUDA_ERROR(err, "Failed to allocate device vector IN");

	int *d_OUT = NULL;
	err = cudaMalloc((void**) &d_OUT, size);
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

// copy the memory from host to device
	err = cudaMemcpy(d_IN, h_IN, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array IN from host to device");
	int blocksPerGridLevel1 = 1 + ((numElements - 1) / (BLOCK_SIZE * 2));

	cudaEventRecord(d_start, 0);
	block_scan_full_BCAO<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT,
			numElements,
			NULL, 0);

	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	printf("Block scan with single thread of %d elements took = %.f5mSecs\n",
			numElements, d_msecs);

	err = cudaMemcpy(h_OUT_CUDA, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array OUT from device to host");

	//// cleanup
	// Free device global memory
	err = cudaFree(d_IN);
	CUDA_ERROR(err, "Failed to free device vector A");
	err = cudaFree(d_OUT);
	CUDA_ERROR(err, "Failed to free device vector B");

	compare_results(h_OUT, h_OUT_CUDA, numElements);
////
//	full_block_scan_BCAO(h_IN, h_OUT_CUDA, numElements);
//
//	compare_results(h_OUT, h_OUT_CUDA, numElements);

// Free host memory
	free(h_IN);
	free(h_OUT);
	free(h_OUT_CUDA);

// Clean up the Host timer
	sdkDeleteTimer(&timer);

// Clean up the Device timer event objects
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	return 0;

}
