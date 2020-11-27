#include "plane-ransac.hpp"
#include "common.hpp"
#include <stdlib.h>

//These should definitely move to common.hpp
__device__ float getX(GPU_Cloud pc, int index) {
    return pc.data[pc.stride * index + 0];
}

__device__ float getY(GPU_Cloud pc, int index) {
    return pc.data[pc.stride * index + 1];
}

__device__ float getZ(GPU_Cloud pc, int index) {
    return pc.data[pc.stride * index + 2];
}

__device__ float3 getPoint(GPU_Cloud pc, int idx) {
    return float3(getX(pc, idx), getY(pc, idx), getZ(pc, idx));
}

/* 
REQUIRES:
    - GPU data cloud
    - A buffer to write inlier counts for each attempted model
    - A buffer to write the randomly selected points for each attempted model
EFFECTS:

Block: 
    Each block represents an "iteration" of the traditional RANSAC algorithm. 
    That is, every block has a different set of randomly chosen 3 points to 
    evaluate the model with. 
Thread:
    Threads are used to decide how many points are inliers to the model. If
    thread max = 1024 and there are 2048 points, each thread will process 2
    points. Each thread will right the number of inliers to its specific spot
    in shared memory. Threads are synced, and then threads will participate
    in a parallel reduction to give the total number of inliers, which will 
    be returned from the kernel along with the three points that define the
    model tested on this iteration.

*/
__global__ void ransacKernel(GPU_Cloud pc, int* inlierCounts, int* modelPoints) {
    int iteration = blockIdx.x;

    int randIdx0 = modelPoints[iteration*3 + 0];
    int randIdx1 = modelPoints[iteration*3 + 1];
    int randIdx2 = modelPoints[iteration*3 + 2];
    float3 modelPt0 = getPoint(pc, randIdx0);
    float3 modelPt1 = getPoint(pc, randIdx1);
    float3 modelPt2 = getPoint(pc, randIdx2);

    int pointsPerThread = (pc.size + MAX_THREADS - 1) / MAX_THREADS;
    for(int i = 0; i < pointsPerThread; i++) {
        int pointIdx = threadIdx.x * pointsPerThread + i;
        if(pointIdx > pc.size) return; 

        float3 curPt = getPoint(pc, pointIdx);

        //calculate distance of cur pt to the plane formed by the 3 model points

        //add a 0 if inlier, 1 if not 
    }
    
    //parallel reduction

    if(threadIdx.x == 0) {
        inlierCounts[iteration] = __shar
    }
}


RansacPlane::RansacPlane(Vector3d axis, float epsilon, int iterations, float threshold, int pcSize)
: pc(pc), axis(axis), epsilon(epsilon), iterations(iterations), threshold(threshold)  {
    //Set up buffers needed for RANSAC
    cudaMalloc(&inlierCounts, sizeof(int) * iterations);
    cudaMalloc(&modelPoints, sizeof(int) * iterations * 3);
    
    //Generate random numbers in CPU to use in RANSAC kernel
    int* randomNumsCPU = (int*) alloc(sizeof(int) * iterations* 3);
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
    ransacKernel<<<blocks, threads>>>(pc, inlierCounts, modelPoints);
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