#pragma once
#include "euclidean-cluster.hpp"
#include <string>

//Helper functions
__device__ float getFloatData(int axis, sl::float4 &val) {
    if(!axis)
        return val.x;
    else if(axis == 1)
        return val.y;
    else
        return val.z;
}
            
__device__ float getData(int axis, int index, sl::float4 *data) {
    return getFloatData(axis, data[index]);    
}


/**
This kernel uses parallel reduction to find the 6 maximum and minimum points
in the point cloud
*/
__global__ void findBoundingBoxKernel(GPU_Cloud_F4 pc, int *minXGlobal, int *maxXGlobal,
                                int *minYGlobal, int *maxYGlobal, int *minZGlobal, int *maxZGlobal){
    //Would it be better to do multiple parallel reductions than one large memory consuming reduction?
    //This method makes 6 copies of the point cloud to find the necessary values 
    const int threads = 8;
    __shared__ int localMinX[threads/2];
    __shared__ int localMaxX[threads/2];
    __shared__ int localMinY[threads/2];
    __shared__ int localMaxY[threads/2];
    __shared__ int localMinZ[threads/2];
    __shared__ int localMaxZ[threads/2];
    __shared__ sl::float4 data[threads];
    __shared__ bool notFull;

    sl::float4 defaultInit(-1.0,-1.0 , -1.0, 0);

    int actualIndex = threadIdx.x + blockIdx.x * blockDim.x;
    

    if(actualIndex < pc.size){ //only write to shared memory if threads about to die
        data[threadIdx.x] = pc.data[actualIndex]; //Write from global memory into shared memory
    }
    else { //Accounts for final block with more threads than points
        notFull = true;
        data[threadIdx.x] = defaultInit;
    }
    __syncthreads();
    printf("(%d, %.1f) ", actualIndex, data[threadIdx.x].x);
    __syncthreads(); //At this point all data has been populated
    if(threadIdx.x == 0) {
        printf("\n %d \n", notFull);
    
    }
    __syncthreads();

    int aliveThreads = threads / 2;

    if(!notFull) { //Don't have to worry about checking for going out of bounds
    
        int minX = threadIdx.x, maxX = minX, minY = minX,
        maxY = minX, minZ = minX, maxZ = minZ; //initialize local indices of mins and maxes
        
        //Hard coding first iteration in order to save memory
        if (threadIdx.x < aliveThreads) {
            minX = (data[aliveThreads + threadIdx.x].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
            maxX = (data[aliveThreads + threadIdx.x].x > data[maxX].x) ? aliveThreads + threadIdx.x : maxX;
            minY = (data[aliveThreads + threadIdx.x].y < data[minY].y) ? aliveThreads + threadIdx.x : minY;
            maxY = (data[aliveThreads + threadIdx.x].y > data[maxY].y) ? aliveThreads + threadIdx.x : maxY;
            minZ = (data[aliveThreads + threadIdx.x].z < data[minZ].z) ? aliveThreads + threadIdx.x : minZ;
            maxZ = (data[aliveThreads + threadIdx.x].z > data[maxZ].z) ? aliveThreads + threadIdx.x : maxZ;
            printf("(%d, %.1f) ", actualIndex, data[minX].x);
            if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                localMinX[threadIdx.x] = minX;
                localMaxX[threadIdx.x] = maxX;
                localMinY[threadIdx.x] = minY;
                localMaxY[threadIdx.x] = maxY;
                localMinZ[threadIdx.x] = minZ;
                localMaxZ[threadIdx.x] = maxZ;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0) printf("\n");
        aliveThreads /= 2;
        __syncthreads();

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
                maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x) ? aliveThreads + threadIdx.x : maxX;
                minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y) ? aliveThreads + threadIdx.x : minY;
                maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y) ? aliveThreads + threadIdx.x : maxY;
                minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z) ? aliveThreads + threadIdx.x : minZ;
                maxZ = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z) ? aliveThreads + threadIdx.x : maxZ;
                printf("(%d, %.1f) ", actualIndex, data[minX].x);
                if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                    localMinX[threadIdx.x] = minX;
                    localMaxX[threadIdx.x] = maxX;
                    localMinY[threadIdx.x] = minY;
                    localMaxY[threadIdx.x] = maxY;
                    localMinZ[threadIdx.x] = minZ;
                    localMaxZ[threadIdx.x] = maxZ;
                }
            }
            __syncthreads();
            aliveThreads /= 2;
            if(threadIdx.x == 0) printf("\n");
            __syncthreads();
        }
    }
    
    else{ //Now we have to worry ab having a block that's not full
        int minX = actualIndex < pc.size ? threadIdx.x : -1, maxX = minX, minY = minX,
        maxY = minX, minZ = minX, maxZ = minZ; //initialize local indices of mins and maxes checking for those exceeding size
        
        //initialize local indices of mins and maxes checking for those exceeding size
        //printf("(%d, %d) ", actualIndex, minX);
        //Hard coding first iteration in order to save memory
        if (threadIdx.x < aliveThreads) {
            
            if(aliveThreads + threadIdx.x + blockDim.x*blockIdx.x < pc.size) { //If points to valid data
                minX = (data[aliveThreads + threadIdx.x].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
                maxX = (data[aliveThreads + threadIdx.x].x > data[maxX].x) ? aliveThreads + threadIdx.x : maxX;
                minY = (data[aliveThreads + threadIdx.x].y < data[minY].y) ? aliveThreads + threadIdx.x : minY;
                maxY = (data[aliveThreads + threadIdx.x].y > data[maxY].y) ? aliveThreads + threadIdx.x : maxY;
                minZ = (data[aliveThreads + threadIdx.x].z < data[minZ].z) ? aliveThreads + threadIdx.x : minZ;
                maxZ = (data[aliveThreads + threadIdx.x].z > data[maxZ].z) ? aliveThreads + threadIdx.x : maxZ;
            }
            printf("(%d, %d) ", actualIndex, minX);
            if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                localMinX[threadIdx.x] = minX;
                localMaxX[threadIdx.x] = maxX;
                localMinY[threadIdx.x] = minY;
                localMaxY[threadIdx.x] = maxY;
                localMinZ[threadIdx.x] = minZ;
                localMaxZ[threadIdx.x] = maxZ;
            }
        }
        __syncthreads();
        if(threadIdx.x == 0) printf("\n");
        aliveThreads /= 2;
        __syncthreads();

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                if(localMinX[aliveThreads + threadIdx.x] >= 0) { //If valid value compare and choose appropriately
                    if(data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) minX = localMinX[aliveThreads + threadIdx.x];
                
                    //minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
                    maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x) ? localMaxX[aliveThreads + threadIdx.x] : maxX;
                    minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y) ? localMinY[aliveThreads + threadIdx.x] : minY;
                    maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y) ? localMaxY[aliveThreads + threadIdx.x] : maxY;
                    minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z) ? localMinY[aliveThreads + threadIdx.x] : minZ;
                    maxZ = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z) ? localMaxZ[aliveThreads + threadIdx.x] : maxZ;
                }
                if (threadIdx.x >= (aliveThreads) / 2) {//Your going to die next iteration, so write to shared
                    localMinX[threadIdx.x] = minX;
                    localMaxX[threadIdx.x] = maxX;
                    localMinY[threadIdx.x] = minY;
                    localMaxY[threadIdx.x] = maxY;
                    localMinZ[threadIdx.x] = minZ;
                    localMaxZ[threadIdx.x] = maxZ;
                }
            }
            __syncthreads();
            if(localMinX[aliveThreads + threadIdx.x] >= 0 && threadIdx.x < aliveThreads)
                printf("(%d, %d) ", actualIndex, minX);
            if(threadIdx.x == 0) printf("\n");
            aliveThreads /= 2;
            __syncthreads();
        }
    }

    //Write to global memory
    if(threadIdx.x == 0){
        minXGlobal[blockIdx.x] = localMinX[0] + blockDim.x*blockIdx.x;
        maxXGlobal[blockIdx.x] = localMaxX[0] + blockDim.x*blockIdx.x;
        minYGlobal[blockIdx.x] = localMinY[0] + blockDim.x*blockIdx.x;
        maxYGlobal[blockIdx.x] = localMaxY[0] + blockDim.x*blockIdx.x;
        minZGlobal[blockIdx.x] = localMinZ[0] + blockDim.x*blockIdx.x;
        maxZGlobal[blockIdx.x] = localMaxZ[0] + blockDim.x*blockIdx.x;
        printf("Smallest X: %.1f at %d\n", data[localMinX[0]].x, localMinX[0]+blockDim.x*blockIdx.x);
        printf("Smallest Y: %.1f at %d\n", data[localMinY[0]].y, localMinY[0]+blockDim.x*blockIdx.x);
        printf("Smallest Z: %.1f at %d\n", data[localMinZ[0]].z, localMinZ[0]+blockDim.x*blockIdx.x);   
    }
    return;

}

/**
The final reduction to extrema to find the ultimate extrema from the
provided list. Split into 3 blocks each calculating the max and min 
values for their given axis. Needed to divide it up since float4 = 16bytes
and we have 2048 float4
*/
__global__ void findExtremaKernel (GPU_Cloud_F4 pc, int size, int *minGlobal, int *maxGlobal, int axis) {
    
    //Copy from global to shared memory
    const int threads = 8;
    __shared__ int localMin[threads];
    __shared__ int localMax[threads];
    __shared__ sl::float4 localMinData[threads];
    __shared__ sl::float4 localMaxData[threads];
    
    //Copy in all of the local data check for uninitialized values
    //Shouldn't cause warp divergence since the first set of contiguous
    //numbers will enter the else and the second half will enter the if
    
    if(threadIdx.x >= size) {
        localMin[threadIdx.x] = -1;
        localMax[threadIdx.x] = -1;
        localMinData[threadIdx.x] = pc.data[0];
        localMaxData[threadIdx.x] = pc.data[0];
    }
    else {
        localMin[threadIdx.x] = minGlobal[threadIdx.x];
        localMax[threadIdx.x] = maxGlobal[threadIdx.x];
        localMinData[threadIdx.x] = pc.data[localMin[threadIdx.x]];
        localMaxData[threadIdx.x] = pc.data[localMax[threadIdx.x]];
    }
    __syncthreads();

    printf("(%d ,%d) ", threadIdx.x, localMin[threadIdx.x]);

    //Registry memory initializations
    int min = localMin[threadIdx.x];
    int max = localMax[threadIdx.x];
    int aliveThreads = (blockDim.x) / 2;
    sl::float4 minData = localMinData[threadIdx.x];
    sl::float4 maxData = localMaxData[threadIdx.x];

    __syncthreads();

    //Do parallel reduction and modify both values as you go along
    while (aliveThreads > 0) {
        if (threadIdx.x < aliveThreads && localMin[threadIdx.x+aliveThreads] != -1) {
            //Check if value smaller than min
            if(getFloatData(axis, minData) > getFloatData(axis, localMinData[threadIdx.x + aliveThreads])) {
                minData = localMinData[threadIdx.x + aliveThreads];
                min = localMin[threadIdx.x + aliveThreads];
            }
            //Check if value larger than max
            if(getFloatData(axis, maxData) < getFloatData(axis, localMaxData[threadIdx.x + aliveThreads])) {
                maxData = localMaxData[threadIdx.x + aliveThreads];
                max = localMax[threadIdx.x + aliveThreads];
            }

            //Check if thread is going to die next iteration
            if (threadIdx.x >= (aliveThreads) / 2) {
                localMin[threadIdx.x] = min;
                localMax[threadIdx.x] = max;
                localMinData[threadIdx.x] = minData;
                localMaxData[threadIdx.x] = maxData;
            }
        }
        __syncthreads();
        if (threadIdx.x < aliveThreads && localMin[threadIdx.x+aliveThreads] != -1){
                printf("(%d, %d) ", threadIdx.x, min);
        }
            if(threadIdx.x == 0) printf("\n");
                aliveThreads /= 2;
            __syncthreads();
    }

    //If final thread write to global memory
    if(threadIdx.x == 0){
        minGlobal[0] = localMin[threadIdx.x];
        maxGlobal[0] = localMax[threadIdx.x];
        std::printf("Axis %i min index: %d\n", axis, localMin[threadIdx.x]);
        std::printf("Axis %i max index: %d\n", axis, localMax[threadIdx.x]);
    }
      
}

/*
Finds the 6 maximum and minimum points needed to define a bounding box around the 
point cloud. Performs a function 6 times to find each point. The maximum pc size
for this function is 1048576 since it assumes the resulting reduction fits into a block
*/
void EuclideanClusterExtractor::findBoundingBox(GPU_Cloud_F4 &pc){
    const int threads = 8;
    int blocks = ceilDiv(pc.size,threads);
    int *minX; //Stores max and min x,y,z values for each block in global memory
    int *maxX;
    int *minY; 
    int *maxY;
    int *minZ; 
    int *maxZ;
    checkStatus(cudaMalloc(&minX, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&maxX, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&minY, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&maxY, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&minZ, sizeof(int) * blocks));
    checkStatus(cudaMalloc(&maxZ, sizeof(int) * blocks));
    
    std::cerr << "blocks: " << blocks << "\n";
    std::cerr << "threads: " << threads << "\n";
    findBoundingBoxKernel<<<blocks,threads>>>(pc, minX, maxX, minY, maxY, minZ, maxZ); //Find 6 bounding values for all blocks
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    std::cerr << "checked 1\n";
    findExtremaKernel<<<1, threads>>>(pc, blocks, minX, maxX, 0);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    std::cerr << "checked 2\n";
    findExtremaKernel<<<1, threads>>>(pc, blocks, minY, maxY, 1);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    std::cerr << "checked 3\n";
    findExtremaKernel<<<1, threads>>>(pc, blocks, minZ, maxZ, 2);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    std::cerr << "checked 4\n";

}
/*
This kernel determines the structure of the graph but does not build it
In theory, there is a memory-compute trade off to be made here. This kernel
is not strictly necessary if we allow an upper bound of memory so that each 
point can have the entire dataset amount of neighbors. Perhaps we can 
explore this allocation method instead.
*/
//b: enough, t: each point
/*
__global__ determineGraphStructureKernel(GPU_Cloud_F4 pc, float tolerance, int* listStart) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    
    //horrible slow way of doing this that is TEMPORARY --> please switch to radix sorted bins
    for(int i = 0; i < pc.size; i++) {
        sl::float3 dvec = (pt - pc.data[i]);
        //this is a neighbor
        if( dvec.norm() < tolerance) {
            neighborCount++;
        }
    }
    listStart[ptIdx] = neighborCount;

    //we must do an exclusive scan using thrust after this kernel
}


 This kernel builds the graph 
Fairly standard adjacency list structure. 

__global__ buildGraphKernel(GPU_Cloud_F4 pc, float tolerance, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    //get the adjacency list for this point
    float* list = neighborLists + listStart[ptIdx]
    
    //horrible slow way of doing this that is TEMPORARY --> please switch to radix sorted bins
    for(int i = 0; i < pc.size; i++) {
        sl::float3 dvec = (pt - pc.data[i]);
        //this is a neighbor
        if( dvec.norm() < tolerance) {
            list[neighborCount] = i;
            neighborCount++;
        }
    }
    
    listStart[ptIdx] = neighborCount;
    labels[ptIdx] = ptIdx;
    f1[ptIdx] = true;
    f2[ptIdx] = false;
}

//this kernel propogates labels, it must be called in a loop
__global__ propogateLabels(GPU_Cloud_F4 pc, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2, bool* m) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;

    bool mod = false;

    if(f1[ptIdx]) {
        float* list = neighborLists + listStart[ptIdx]
        int listLen = listStart[ptIdx+1] - listStart[ptIdx];
        f1[ptIdx] = false;
        int myLabel = labels[ptIdx];

        for(int i = 0; i < listLen; i++) {
            int otherLabel = labels[list[i]];
            if(myLabel < otherLabel) { //are these reads actually safe?
                atomicMin(&labels[list[i]], myLabel);
                f2[list[i]] = true;
                *m = true;
            } else {
                myLabel = otherLabel;
                mod = true;
            }
        }
    }

    if(mod) {
        atomicMin(&labels[ptIdx], myLabel);
        f2[ptIdx] = true;
        *m = true
    }
}

//this debug kernel colors points based on their label
__global__ colorClusters(GPU_Cloud_F4 pc, int* labels) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx == )
}
*/
EuclideanClusterExtractor::EuclideanClusterExtractor(float tolerance, int minSize, float maxSize) 
: tolerance(tolerance), minSize(minSize), maxSize(maxSize) {}
/*
EuclideanClusterExtractor::extractClusters(GPU_Cloud_F4 pc) {

}
*/