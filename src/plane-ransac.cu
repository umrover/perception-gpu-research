#include "plane-ransac.hpp"
#include "common.hpp"
#include <stdlib.h>

//These should definitely move to common.hpp but uncear on how to put device functions in header
__device__ float getX(GPU_Cloud &pc, int index) {
    return pc.data[pc.stride * index + 0];
}

__device__ float getY(GPU_Cloud &pc, int index) {
    return pc.data[pc.stride * index + 1];
}

__device__ float getZ(GPU_Cloud &pc, int index) {
    return pc.data[pc.stride * index + 2];
}

__device__ float3 getPoint(GPU_Cloud &pc, int idx) {
    return make_float3(getX(pc, idx), getY(pc, idx), getZ(pc, idx));
}

__device__ float3 cross(float3 &a, float3 &b) { 
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ float dot(float3 &a, float3 &b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float norm(float3 &a) {
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

__device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

__device__ float3 operator/(const float3 &a, const float b) {
    return make_float3(a.x/b, a.y/b, a.z/b);
}


__device__ int ceilDivGPU(int a, int b) {
    return (a + b - 1) / b;
}

/* 
LAUNCH:
    - [Block] # iterations [aka, randomly selected models] to try
    - [Thread] MAX_THREADS

REQUIRES:
    - GPU point cloud
    - A buffer to write inlier counts for each attempted model
    - A buffer that tells the kernel what the randomly selected points were for each model
    - Threshold distance for a pt to be considered an inlier
EFFECTS:

Block: 
    Each block represents an "iteration" of the traditional RANSAC algorithm. 
    That is, every block has a different set of randomly chosen 3 points that define the
    model (a plane is minimally defined by 3 points). The threads in the block then check
    every point in the point cloud against the model for that block.
Thread:
    Threads are used to decide how many points are inliers to the model. If
    thread max = 1024 and there are 2048 points, each thread will process 2
    points. Each thread will write the number of inliers in the set of points it evaluated
    to its specific spot in shared memory. Threads are synced, and then threads will 
    participate in a parallel reduction to give the total number of inliers for that block/model, 
    which will be returned from the kernel in the inlierCounts buffer. 
*/
__global__ void ransacKernel(GPU_Cloud pc, int* inlierCounts, int* modelPoints, int threshold) {
    __shared__ int inlierField[MAX_THREADS];
    //inlierField[threadIdx.x] = 0;
    int inliers = 0;
    int iteration = blockIdx.x;

    // select 3 random points from the cloud as the model that this particular block will evaluate
    int randIdx0 = modelPoints[iteration*3 + 0];
    int randIdx1 = modelPoints[iteration*3 + 1];
    int randIdx2 = modelPoints[iteration*3 + 2];
    float3 modelPt0 = getPoint(pc, randIdx0);
    float3 modelPt1 = getPoint(pc, randIdx1);
    float3 modelPt2 = getPoint(pc, randIdx2);

    // figure out how many points each thread must compute distance for and determine if each is inlier/outlier
    int pointsPerThread = ceilDivGPU(pc.size, MAX_THREADS);
    for(int i = 0; i < pointsPerThread; i++) {
        // select a point index or return if this isn't a valid point
        int pointIdx = threadIdx.x * pointsPerThread + i;
        if(pointIdx > pc.size) return; 
        
        // point in the point cloud that could be an inlier or outlier
        float3 curPt = getPoint(pc, pointIdx);

        // get the two vectors on the plane defined by the model points
        float3 v1 = modelPt1 - modelPt0;
        float3 v2 = modelPt2 - modelPt0;

        //get a vector normal to the plane model
        float3 n = cross(v1, v2);

        //calculate distance of cur pt to the plane formed by the 3 model points [see doc for the complete derrivation]
        float3 d_to_model_pt = (curPt - modelPt1);
        float d = abs(dot(n, d_to_model_pt)) / norm(n);

        //add a 0 if inlier, 1 if not 
        inliers += (d < threshold) ? 1 : 0; 
    }
    
    //parallel reduction to get an aggregate sum of the number of inliers for this model
    //this is all equivalent to sum(inlierField), but it does it in parallel
    inlierField[threadIdx.x] = inliers;
    __syncthreads();
    int aliveThreads = (blockDim.x) / 2;
	while (aliveThreads > 0) {
		if (threadIdx.x < aliveThreads) {
			inliers += inlierField[aliveThreads + threadIdx.x];
			if (threadIdx.x >= (aliveThreads) / 2) inlierField[threadIdx.x] = inliers;
		}
		__syncthreads();
		aliveThreads /= 2;
	}

    //at the final thread, write to global memory
    if(threadIdx.x == 0) {
        inlierCounts[iteration] = inliers;
    }
}


RansacPlane::RansacPlane(Vector3d axis, float epsilon, int iterations, float threshold, int pcSize)
: pc(pc), axis(axis), epsilon(epsilon), iterations(iterations), threshold(threshold)  {
    //Set up buffers needed for RANSAC
    cudaMalloc(&inlierCounts, sizeof(int) * iterations);
    cudaMalloc(&modelPoints, sizeof(int) * iterations * 3);
    
    //Generate random numbers in CPU to use in RANSAC kernel
    int* randomNumsCPU = (int*) malloc(sizeof(int) * iterations* 3);
    for(int i = 0; i < iterations*3; i++) {
        randomNumsCPU[i] = rand() % pcSize;
    }
    cudaMemcpy(modelPoints, randomNumsCPU, sizeof(int) * iterations * 3, cudaMemcpyHostToDevice);
    free(randomNumsCPU);
}

/*  
EFFECTS:
    1. [GPU] Use the RANSAC kernel to evaluate all the canidate models and report their associated inlier count
    2. [GPU] Select the canidate with the highest score and copy its three model points back to the CPU
    3. [CPU] Use the three model points to produce a plane equation in standard form and return to the user
*/
RansacPlane::Plane RansacPlane::computeModel(GPU_Cloud pc) {
    this->pc = pc;

    int blocks = iterations;
    int threads = MAX_THREADS;
    ransacKernel<<<blocks, threads>>>(pc, inlierCounts, modelPoints, threshold);
    checkStatus(cudaGetLastError());
    checkStatus(cudaDeviceSynchronize());

    return {0, 0, 0, 0};
}

/*
EFFECTS:
    1. Uses the selection computed in computeModel() and the modelPoints of that selection
    to re-calculate the inliers and export them in a list. 
*/
GPU_Indicies RansacPlane::getInliers() {
    return {nullptr, 0};
}


RansacPlane::~RansacPlane() {
    cudaFree(inlierCounts);
    cudaFree(modelPoints);
}