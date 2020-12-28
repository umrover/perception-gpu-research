#include "test-filter-f4.hpp"
#include <iostream>

#include "common.hpp"

//Cuda kernel that turns pixels blue in parallel on the GPU!
__global__ void blueKernel(GPU_Cloud_F4 cloud) {
    int idx = threadIdx.x + blockIdx.x * 1024;
    if(idx >= cloud.size) return;
    
    //Set the color of all points to blue
    //cloud.data[idx*4+3] = 4353.0;
    
    
    //cloud.data[idx].w = 9932999;/// cloud.data[idx].norm();
    sl::float3 v3(cloud.data[idx]);
    cloud.data[idx].w = 4353.0;//v3.norm();
    
}

//Constructor for test filter, take in a ZED GPU cloud and convert to a struct that can be passed to CUDA Kernels
TestFilter_F4::TestFilter_F4(sl::Mat gpu_cloud) {
    this->gpu_cloud.data = gpu_cloud.getPtr<sl::float4>(sl::MEM::GPU);
    this->gpu_cloud.stride = 1;
    this->gpu_cloud.size = gpu_cloud.getWidth() * gpu_cloud.getHeight();
}

//Run the filter on the point cloud
void TestFilter_F4::run() {
    blueKernel<<<ceilDiv(gpu_cloud.size, 1024), 1024>>>(this->gpu_cloud);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
}