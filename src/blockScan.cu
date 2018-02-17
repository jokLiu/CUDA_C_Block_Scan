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

#define BLOCK_SIZE 1024

// for avoiding bank conflicts
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
		((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
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
void block_scan_BCAO(int *g_idata, int *g_odata, int n) {
	__shared__ int temp[BLOCK_SIZE * 8];  // allocated on invocation

	int thid = threadIdx.x;
	int offset = 1;

	// ADDED OUR MODIFIED
	int ai = thid;
	int bi = thid + (n / 2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
	// ADDED OUR MODIFIED END

	for (int d = n >> 1; d > 0; d >>= 1)       // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			// ADDED OUR MODIFIED
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			// ADDED OUR MODIFIED END
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// ADDED OUR MODIFIED
	if (thid==0) temp[n-1 + CONFLICT_FREE_OFFSET(n-1)] = 0;
	// ADDED OUR MODIFIED END

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
			{
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			// ADDED OUR MODIFIED
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			// ADDED OUR MODIFIED END
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	// ADDED OUR MODIFIED
	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];
	// ADDED OUR MODIFIED END
}

__global__
void block_scan(int *g_idata, int *g_odata, int n) {
	__shared__ int temp[BLOCK_SIZE * 2];  // allocated on invocation

	int thid = threadIdx.x;
	int offset = 1;

	temp[2 * thid] = g_idata[2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[2 * thid + 1];

	for (int d = n >> 1; d > 0; d >>= 1)       // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) {
		temp[n - 1] = 0;
	} // clear the last element

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
			{
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
}

__global__
void block_scan_full(int *g_idata, int *g_odata, int n, int *SUM,
		int add_last) {
	__shared__ int temp[BLOCK_SIZE * 2];  // allocated on invocation

	int thid = threadIdx.x;
	int blockId = blockDim.x * blockIdx.x * 2;
	int offset = 1;
	int last = 0;

	temp[2 * thid] = g_idata[blockId + 2 * thid]; // load input into shared memory
	temp[2 * thid + 1] = g_idata[blockId + 2 * thid + 1];

	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
	/*SUM[blockIdx.x] */
		last = temp[2 * thid + 1];

	for (int d = BLOCK_SIZE /*n >> 1*/; d > 0; d >>= 1) // build sum in place up the tree
			{
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) {
		temp[BLOCK_SIZE * 2 - 1] = 0;
	} // clear the last element

	for (int d = 1; d < BLOCK_SIZE * 2; d *= 2) // traverse down tree & build scan
			{
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
	if (add_last && thid == BLOCK_SIZE - 1) // save the last element for later
		SUM[blockIdx.x] = temp[2 * thid + 1] + last;
	g_odata[blockId + 2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[blockId + 2 * thid + 1] = temp[2 * thid + 1];
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
	err = cudaMalloc((void**) &d_SUM_1, (int) ceil(len / (BLOCK_SIZE << 1)));
	CUDA_ERROR(err, "Failed to allocate device vector OUT");

	int *d_SUM_1_Scanned = NULL;
	err = cudaMalloc((void**) &d_SUM_1_Scanned,
			(int) ceil(len / (BLOCK_SIZE << 1)));
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
	int blocksPerGridLevel2 = ceil(blocksPerGridLevel1 / (BLOCK_SIZE << 1));

	printf("\n%d\n\n", blocksPerGridLevel2);

	cudaEventRecord(d_start, 0);
	block_scan_full<<<blocksPerGridLevel1, BLOCK_SIZE>>>(d_IN, d_OUT, len,
			d_SUM_1, 1);
	block_scan_full<<<blocksPerGridLevel2, BLOCK_SIZE>>>(d_SUM_1,
			d_SUM_1_Scanned, blocksPerGridLevel2, d_SUM_2, 1);
	block_scan_full<<<1, BLOCK_SIZE>>>(d_SUM_2, d_SUM_2_Scanned,
	BLOCK_SIZE << 1, NULL, 0);
//	 TODO check if we need that -1
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
	int numElements = 2048;
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
	sequential_scan(h_IN, h_OUT, numElements);

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

	cudaEventRecord(d_start, 0);
	block_scan_BCAO<<<1, BLOCK_SIZE>>>(d_IN, d_OUT, numElements);
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
//
//	full_block_scan(h_IN, h_OUT_CUDA, numElements);
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
