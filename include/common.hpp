#include <sl/Camera.hpp>

#ifndef COMMON
#define COMMON

#define CHANNEL 4
#define BLOCK_SIZE 1024

//GPU point cloud struct that can be passed to cuda kernels and represents a point cloud
struct GPU_Cloud {
    float* data;
    int stride; 
    int size;
};

/**
 \brief: returns data corresponding to axis
 \param axis: 0 is x, 1 is y, 2 is z
 */
float float4::getData(int axis);

struct GPU_Cloud_F4 {
    sl::float4* data;
    int stride; 
    int size;

    /**
     \brief returns data corresponding to axis and index 
     \param axis: 0 is x, 1 is y, 2 is z 
     */
    float getData(int axis, int index);
};

//GPU Indicies data
struct GPU_Indicies {
    int* data;
    int size;
};


//Returns true if a cuda error occured and prints an error message
bool checkStatus(cudaError_t status);

//ceiling division x/y. e.g. ceilDiv(3,2) -> 2
int ceilDiv(int x, int y);

//Get a CUDA workable gpu point cloud struct from Zed GPU cloud
GPU_Cloud getRawCloud(sl::Mat zed_cloud);

GPU_Cloud_F4 getRawCloud(sl::Mat zed_cloud, bool f4);


//Remove all the points in cloud except those at the given indicies 
GPU_Cloud removeAllExcept(GPU_Cloud pc, GPU_Indicies indicies);

//Remove all the points in cloud at the given indicies 
GPU_Cloud keepAllExcept(GPU_Cloud pc, GPU_Indicies indicies);

/*
__device__ float getX(GPU_Cloud pc, int index) {
    return pc.data[pc.stride * index + 0];
}

__device__ float getY(GPU_Cloud pc, int index) {
    return pc.data[pc.stride * index + 1];
}

__device__ float getZ(GPU_Cloud pc, int index) {
    return pc.data[pc.stride * index + 2];
} */
#define MAX_THREADS 1024

#endif