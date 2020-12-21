#include "pass-through.hpp"
#include <stdlib.h>
#include <cmath>
#include <limits>

PassThrough::PassThrough(char axis, float min, float max) : min{min}, max{max} {

    //Set the axis value
    if(axis == 'x')
        this->axis = 0;
    else if(axis == 'y')
        this->axis = 1;
    else
        this->axis = 2;

};

//CUDA Kernel Helper Function
__global__ void passThroughKernel(GPU_Cloud_F4 cloud, int axis, float min, float max, int* size) {

    //Find index for current operation
    int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if(idx >= cloud.size) return;
    //printf("index X z value = %f", cloud.data[idx].z);
    //If out of range make blue and return
    if(cloud.data[idx].z > max || cloud.data[idx].z < min ||
        isnan(cloud.data[idx].z) || isinf(cloud.data[idx].z)){
        cloud.data[idx].w = 4353.0;
        return;
    }
    
    //If still going then update point cloud float array
    sl::float4 copy = cloud.data[idx];

    //Make sure all threads have checked for passThrough
    __syncthreads();

    //Count the new size
    int place = atomicAdd(size, 1);
    //printf("Atomic added\n");
    //Copy back data into place in front of array
    cloud.data[place] = copy;

}

void PassThrough::run(GPU_Cloud_F4 cloud){

    std::cerr << "Original size: " << cloud.size << "\n";
    
    //Create pointer to value in host memory
    int* h_newSize = new int;
    *h_newSize = 0;

    //Create pointer to value in device memory
    int* d_newSize = nullptr;
    cudaMalloc(&d_newSize, sizeof(int));

    //Copy from host to device
    cudaMemcpy(d_newSize, h_newSize, sizeof(int), cudaMemcpyHostToDevice);
    checkStatus(cudaGetLastError());
    
    //Run PassThrough Kernel
    passThroughKernel<<<ceilDiv(cloud.size, BLOCK_SIZE), BLOCK_SIZE>>>(cloud, axis, min, max, d_newSize);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    
    //Copy from device to host
    cudaMemcpy(h_newSize, d_newSize, sizeof(int), cudaMemcpyDeviceToHost);
    checkStatus(cudaGetLastError());
    
    //Update size of cloud
    cloud.size = *h_newSize;
    //cloud.data[*d_newSize] = nullptr;
    checkStatus(cudaGetLastError());
    std::cerr << "New Cloud Size: " << cloud.size << "\n";

    //Free dynamically allocated memory
    cudaFree(d_newSize);
    delete h_newSize;
}