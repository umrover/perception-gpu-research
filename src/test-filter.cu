#include "test-filter.hpp"
#include <iostream>

#include "common.hpp"

//Cuda kernel that turns pixels blue in parallel on the GPU!
__global__ void blueKernel(GPU_Cloud cloud) {
    int idx = threadIdx.x + blockIdx.x * 1024;
    if(idx >= cloud.size) return;
    
    //Set the color of all points to blue
    cloud.data[idx*4+3] = 4353.0;
}

//Constructor for test filter, take in a ZED GPU cloud and convert to a struct that can be passed to CUDA Kernels
TestFilter::TestFilter(sl::Mat gpu_cloud) {
    this->gpu_cloud.data = gpu_cloud.getPtr<float>(sl::MEM::GPU);
    this->gpu_cloud.stride = 1;
    this->gpu_cloud.size = gpu_cloud.getWidth() * gpu_cloud.getHeight();
}

//Run the filter on the point cloud
void TestFilter::run() {
    blueKernel<<<ceilDiv(gpu_cloud.size, 1024), 1024>>>(this->gpu_cloud);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
}