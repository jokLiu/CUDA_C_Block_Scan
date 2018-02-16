#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}

#define BLOCK_SIZE 1024


static void compare_results(const int *vector1, const int *vector2,
		int numElements) {
	for (int i = 0; i < numElements; ++i) {
		if (fabs(vector1[i] - vector2[i]) > 1e-5f) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}

__host__
void sequential_scan(int *g_idata, int *g_odata, int n) {

	g_odata[0] = 0;
	for(int i=1; i<n; i++){
		g_odata[i] = g_odata[i-1] + g_idata[i-1];
	}
}

__global__
void block_scan(int *g_idata, int *g_odata, int n) {
	__shared__ int temp[BLOCK_SIZE*2];  // allocated on invocation

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
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2 * thid] = temp[2 * thid]; // write results to device memory
	g_odata[2 * thid + 1] = temp[2 * thid + 1];
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
		h_IN[i] = rand() % 10;
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
	block_scan<<<1, BLOCK_SIZE>>>(d_IN, d_OUT, numElements);
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


	// Free host memory
	free (h_IN);
	free (h_OUT);
	free (h_OUT_CUDA);

	// Clean up the Host timer
	sdkDeleteTimer(&timer);

	// Clean up the Device timer event objects
	cudaEventDestroy (d_start);
	cudaEventDestroy (d_stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");



	return 0;

}
