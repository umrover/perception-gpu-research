#pragma once

//Returns true if a cuda error occured
__host__ bool checkStatus(cudaError_t status) {
	if (status != cudaSuccess) {
		printf("%s \n", cudaGetErrorString(status));
		return true;
	}
    return false;
}

//TODO
GPU_Cloud removeAllExcept(GPU_Cloud in, int* indicies) {
    return {nullptr, 0, 0};
}

//TODO
GPU_Cloud keepAllExcept(GPU_Cloud in, int* indicies) {
    return {nullptr, 0, 0};
}

//GPU point cloud struct
struct GPU_Cloud {
    float* data;
    int stride; 
    int size;
};