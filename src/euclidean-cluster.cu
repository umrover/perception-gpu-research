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
    const int threads = MAX_THREADS;
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

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) ? localMinX[aliveThreads + threadIdx.x] : minX;
                maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x) ? localMaxX[aliveThreads + threadIdx.x] : maxX;
                minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y) ? localMinY[aliveThreads + threadIdx.x] : minY;
                maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y) ? localMaxY[aliveThreads + threadIdx.x] : maxY;
                minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z) ? localMinZ[aliveThreads + threadIdx.x] : minZ;
                maxZ = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z) ? localMaxZ[aliveThreads + threadIdx.x] : maxZ;
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
        }
    }
    
    else{ //Now we have to worry ab having a block that's not full
        int minX = actualIndex < pc.size ? threadIdx.x : -1, maxX = minX, minY = minX,
        maxY = minX, minZ = minX, maxZ = minX; //initialize local indices of mins and maxes checking for those exceeding size
        
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

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                if(localMinX[aliveThreads + threadIdx.x] >= 0) { //If valid value compare and choose appropriately
                    if(data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) minX = localMinX[aliveThreads + threadIdx.x];
                
                    //minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x) ? aliveThreads + threadIdx.x : minX;
                    maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x) ? localMaxX[aliveThreads + threadIdx.x] : maxX;
                    minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y) ? localMinY[aliveThreads + threadIdx.x] : minY;
                    maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y) ? localMaxY[aliveThreads + threadIdx.x] : maxY;
                    minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z) ? localMinZ[aliveThreads + threadIdx.x] : minZ;
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
            aliveThreads /= 2;
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
    }
    return;

}

/**
The final reduction to extrema to find the ultimate extrema from the
provided list. Split into 3 blocks each calculating the max and min 
values for their given axis. Needed to divide it up since float4 = 16bytes
and we have 2048 float4
*/
__global__ void findExtremaKernel (GPU_Cloud_F4 pc, int size, int *minGlobal, int *maxGlobal, 
    float* finalMin, float* finalMax, int axis) {
    
    //Copy from global to shared memory
    const int threads = MAX_THREADS;
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
        aliveThreads /= 2;
    }

    //If final thread write to global memory
    if(threadIdx.x == 0){
        finalMin[axis] = getFloatData(axis, minData);
        finalMax[axis] = getFloatData(axis, maxData);
        std::printf("Axis %i min index: %.1f\n", axis, getFloatData(axis, localMinData[threadIdx.x]));
        std::printf("Axis %i max index: %.1f\n", axis, getFloatData(axis, localMaxData[threadIdx.x]));
        
        //If the last axis calculated readjust so the values make a cube
        if(axis == 2){
            float difX = finalMax[0]-finalMin[0];
            float difY = finalMax[1]-finalMin[1];
            float difZ = finalMax[2]-finalMin[2];
    
            if(difZ >= difY && difZ >= difX) {
                int addY = (difZ-difY)/2+1;
                int addX = (difZ-difX)/2+1;
                finalMax[0] += addX;
                finalMin[0] -= addX;
                finalMax[1] += addY;
                finalMin[1] -= addY; 
            }

            else if(difY >= difX && difY >= difZ) {
                int addZ = (difY-difZ)/2+1;
                int addX = (difY-difX)/2+1;
                finalMax[0] += addX;
                finalMin[0] -= addX;
                finalMax[2] += addZ;
                finalMin[2] -= addZ;
            }

            else {
                int addY = (difX-difY)/2+1;
                int addZ = (difY-difZ)/2+1;
                finalMax[2] += addZ;
                finalMin[2] -= addZ;
                finalMax[1] += addY;
                finalMin[1] -= addY;
            }

        }
    }
      
}

/*
Finds the 6 maximum and minimum points needed to define a bounding box around the 
point cloud. Performs a function 6 times to find each point. The maximum pc size
for this function is 1048576 since it assumes the resulting reduction fits into a block
*/
void EuclideanClusterExtractor::findBoundingBox(GPU_Cloud_F4 &pc){
    const int threads = MAX_THREADS;
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
    checkStatus(cudaMalloc(&mins, sizeof(int) * 3));
    checkStatus(cudaMalloc(&maxes, sizeof(int) * 3));

    //Find 6 bounding values for all blocks
    findBoundingBoxKernel<<<blocks,threads>>>(pc, minX, maxX, minY, maxY, minZ, maxZ); 
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Find X extrema in remaining array
    findExtremaKernel<<<1, threads>>>(pc, blocks, minX, maxX, mins, maxes, 0);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Find Y extrema in remaining array
    findExtremaKernel<<<1, threads>>>(pc, blocks, minY, maxY, mins, maxes, 1);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Find Z extrema
    findExtremaKernel<<<1, threads>>>(pc, blocks, minZ, maxZ, mins, maxes, 2);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    //Should print out 2,9,0,7,1,6

    //Free memory
    cudaFree(minX);
    cudaFree(maxX);
    cudaFree(minY);
    cudaFree(maxY);
    cudaFree(minZ);
    cudaFree(maxZ);

}

/*
This kernel will use a hash function to determine which bin the point hashes into
and will then atomically count the number of points to be added to the bin. 
THERE IS DEFINITELY A BETTER WAY TO DO THIS STEP
*/

__global__ void buildBinsKernel(GPU_Cloud_F4 pc, int* binCount, int partitions, 
                                        float* min, float* max, int** bins) {
    
    if(threadIdx.x == 0)
    printf("%i\n", binCount[0]);
    int ptIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if(ptIdx > pc.size) return;

    //Copy Global to registry memory
    sl::float4 data = pc.data[ptIdx];
    if(threadIdx.x == 0)
        printf("minX: %.1f, maxX: %.1f, minY: %.1f, maxY: %.1f, minZ: %.1f, maxZ: %.1f\n", min[0], max[0], min[1], max[1], min[2], max[2]);
    __syncthreads();
    //Use hash function to find bin number
    int cpx = (data.x-min[0])/(max[0]-min[0])*partitions;
    int cpy = (data.y-min[1])/(max[1]-min[1])*partitions;
    int cpz = (data.z-min[2])/(max[2]-min[2])*partitions;
    int binNum = cpx*partitions*partitions+cpy*partitions+cpz;
    
    printf("(%i, %i)", ptIdx, binNum);
    __syncthreads();
    if(threadIdx.x == 0)
        printf("%i\n", binCount[0]);

    int place = atomicAdd(&binCount[binNum],1);
    if(threadIdx.x == 0)
        printf("%i\n", binCount[0]);
    printf("(%i, %i)", ptIdx, place);
    __syncthreads();
    if(threadIdx.x == 0)
        printf("\n");

    //Dynamically allocate memory for bins in kernel. Memory must be freed
    //in a different Kernel. It cannot be freed with cudaFree()
    //By definition of the hash function there will be partitions^3 bins 
    if(ptIdx < partitions*partitions*partitions)
        bins[ptIdx] = (int*)malloc(sizeof(int)*(binCount[ptIdx]));

     // Check for failure
    if (ptIdx < partitions*partitions*partitions && bins[ptIdx] == NULL)
    return;

    __syncthreads();
                                       
    //Memory now exists, so write index to global memory
    bins[binNum][place] = ptIdx;

    __syncthreads();
    if(threadIdx.x == 0 && bins[0] == NULL)
    printf("fish\n");
    printf("(%i, %i, %i), ", binNum, place, bins[binNum][place]);
}

__global__ void freeBinsKernel(int* binCount, int** bins, int partitions){
    
    int ptIdx = threadIdx.x + blockDim.x * blockIdx.x;

    if(threadIdx.x == 0)
    printf("%i\n", binCount[0]);

    //If valid bin
    if(ptIdx < partitions*partitions*partitions){
        int* ptr = bins[ptIdx];
        //If memory was allocated
        if(ptr != NULL)
            free(ptr);
    }
}


/*
This function builds the bins that are needed to prevent an O(n^2) search time
for nearest neighbors. Uses min and max values to construct a cube that can be 
divided up into a specified number of partitions on each axis. 
*/
void EuclideanClusterExtractor::buildBins(GPU_Cloud_F4 &pc) {
    int threads = MAX_THREADS;
    int blocks = ceilDiv(pc.size, threads);
    
    //Allocate memory
    checkStatus(cudaMalloc(&bins, sizeof(int*) * partitions*partitions*partitions));
    checkStatus(cudaMalloc(&binCount, sizeof(int) * partitions*partitions*partitions));
    
    //Construct the bins to be used
    buildBinsKernel<<<blocks, threads>>>(pc, binCount, partitions, mins, maxes, bins);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();
    std::cerr << "buildBins working\n";
}

/*
This function frees dynamically allocated memory in buildBins function
*/
void EuclideanClusterExtractor::freeBins() {
    int threads = MAX_THREADS;
    int blocks = ceilDiv(partitions*partitions*partitions, threads);
    
    freeBinsKernel<<<blocks,threads>>>(binCount, bins, partitions);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    checkStatus(cudaFree(binCount));
    checkStatus(cudaFree(bins));
    std::cerr << "Everythying freed\n";
    
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
EuclideanClusterExtractor::EuclideanClusterExtractor(float tolerance, int minSize, float maxSize, int partitions) 
: tolerance{tolerance}, minSize{minSize}, maxSize{maxSize}, partitions{partitions} {}
/*
EuclideanClusterExtractor::extractClusters(GPU_Cloud_F4 pc) {

}
*/