#include "plane-ransac.hpp"

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

__device__ float getDist() {

}

/* 
REQUIRES:
    - GPU data cloud
    - 
MODIFIES:
    - 

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
    be returned from the kernel. 

*/
__global__ void ransacKernel(GPU_cloud data, float* inlierCount, float* modelIndicies) {

}


RansacPlane::RansacPlane(GPU_Cloud pc, Vector3d axis, float epsilon, int iterations, float threshold)
: pc(pc), axis(axis), epsilon(epsilon), iterations(iterations), threshold(threshold)  {
    
}

// TODO, do steps 2 & 3 in GPU also 
/*
    1. [GPU] Use the RANSAC kernel to evaluate all the canidate models and report their associated inlier count
    2. [GPU] Select the canidate with the highest score
    3. [GPU] Recompute the outliers and inliers for this model 
*/
RansacPlane::Plane RansacPlane::computeModel() {
    dim3 block;
    dim3 grid;
    ransacKernel<<<grid, block>>>(data);
    checkStatus(cudaGetLastError());
    checkStatus(cudaDeviceSynchronize());
}

