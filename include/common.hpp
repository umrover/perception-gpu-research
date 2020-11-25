#ifndef COMMON
#define COMMON


//GPU point cloud struct that can be passed to cuda kernels and represents a point cloud
struct GPU_Cloud {
    float* data;
    int stride; 
    int size;
};

//Returns true if a cuda error occured and prints an error message
bool checkStatus(cudaError_t status);

//ceiling division x/y. e.g. ceilDiv(3,2) -> 2
int ceilDiv(int x, int y);

//Remove all the points in cloud except those at the given indicies 
GPU_Cloud removeAllExcept(GPU_Cloud in, int* indicies, int cnt);

//Remove all the points in cloud at the given indicies 
GPU_Cloud keepAllExcept(GPU_Cloud in, int* indicies, int cnt);


#endif