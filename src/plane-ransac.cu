#include "plane-ransac.hpp"

__global__ void ransacKernel(GPU_cloud data) {

}

RansacPlane::Plane RansacPlane::computeModel() {
    dim3 block;
    dim3 grid;
    ransacKernel<<<grid, block>>>(data);
    checkStatus(cudaGetLastError());

}