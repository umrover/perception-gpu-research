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
__global__ void passThroughKernel(GPU_Cloud cloud, int axis, float min, float max, int* size) {
    
    //Find index for current operation
    int idx = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if(idx >= cloud.size) return;
    
    //Initialize local variables
    float x(0);
    float y(0); 
    float z(0);
    float rgb(0);

    //If out of range make blue and return
    if(cloud.data[idx*CHANNEL+axis] > max || cloud.data[idx*CHANNEL+axis] < min){
        cloud.data[idx*CHANNEL+3] = 4353.0;
        return;
    }
    
    //If still going then update point cloud float array
    x = cloud.data[idx*CHANNEL];
    y = cloud.data[idx*CHANNEL+1];
    z = cloud.data[idx*CHANNEL+2];
    rgb = cloud.data[idx*CHANNEL+3];

    //Make sure all threads have checked for passThrough
    __syncthreads();

    //Count the new size
    int place = atomicAdd(size, 1);

    //Copy back data into place in front of array
    cloud.data[place*CHANNEL] = x;
    cloud.data[place*CHANNEL+1] = y;
    cloud.data[place*CHANNEL+2] = z;
    cloud.data[place*CHANNEL+3] = rgb;

}

void PassThrough::run(GPU_Cloud cloud){

    //Create pointer to value in host memory
    int* h_newSize;
    *h_newSize = 0;

    //Create pointer to value in device memory
    int* d_newSize;
    checkStatus(cudaMalloc(&d_newSize, sizeof(int)));

    //Copy from host to device
    checkStatus(cudaMemcpy(d_newSize, h_newSize, sizeof(int), cudaMemcpyHostToDevice));

    //Run PassThrough Kernel
    passThroughKernel<<<ceilDiv(gpu_cloud.size, BLOCK_SIZE), BLOCK_SIZE>>>(this->gpu_cloud, axis, min, max, d_newSize);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Copy from device to host
    checkStatus(cudaMemcpy(h_newSize, d_newSize, sizeof(int), cudaMemcpyDeviceToHost));

    //Update size of cloud
    cloud.size() = *h_newSize;

}