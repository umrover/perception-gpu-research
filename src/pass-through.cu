#include "pass-through.hpp"
#include "common.hpp"

PassThrough::PassThrough(sl::Mat gpu_cloud, char axis, float min, 
    float max) : min{min} max{max} {

    //Set the axis value
    if(axis == 'x')
        this->axis = 0;
    else if(axis == 'y')
        this->axis = 1;
    else
        this->axis = 2;

    //Initialize the gpu_cloud data member from sl::Mat
    this->gpu_cloud.data = gpu_cloud.getPtr<float>(sl::MEM::GPU);
    this->gpu_cloud.stride = 1;
    this->gpu_cloud.size = gpu_cloud.getWidth() * gpu_cloud.getHeight();
};

//CUDA Kernel Helper Function
__global__ void passThroughKernel(GPU_Cloud cloud, int axis, float min, float max, int* numRemoved) {
    
    //Find index for current operation
    int idx = threadIdx.x + blockIdx.x * 1024;
    if(idx >= cloud.size) return;
    
    //Initialize local variables
    float x(0);
    float y(0); 
    float z(0);
    float rgb(0);

    //If in range set the first value to infinity
    if(cloud.data[idx+axis] > max || cloud.data[idx+axis] < min){
        cloud.data[idx+3] = 4353.0;
        return;
    }
    
    //If still going then update point cloud float array
    x = cloud.data[idx];
    y = cloud.data[idx+1];
    z = cloud.data[idx+2];
    rgb = cloud.data[idx+3];

    //Make sure all threads have checked for passThrough
    __syncthreads();

    //Count the number removed
    int place = atomicAdd(numRemoved, 1);

    //Copy back data into place in front of array
    cloud.data[place] = x;
    cloud.data[place+1] = y;
    cloud.data[place+2] = z;
    cloud.data[place+3] = rgb;

}

void PassThrough::run(GPU_Cloud cloud){

    //Create pointer to value in host memory
    int* h_removed;
    *h_removed = 0;

    //Create pointer to value in device memory
    int* d_removed;
    cudaMalloc(&d_removed, sizeof(int));

    //Copy from host to device
    cudaMemcpy(d_removed, h_removed, sizeof(int), cudaMemcpyHostToDevice);

    //Run PassThrough Kernel
    passThroughKernel<<<ceilDiv(gpu_cloud.size, 1024), 1024>>>(this->gpu_cloud, axis, min, max, d_removed);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Copy from device to host
    cudaMemcpy(h_removed, d_removed, sizeof(int), cudaMemcpyDeviceToHost);

    //Update size of cloud
    cloud.size() -= *h_removed;

}