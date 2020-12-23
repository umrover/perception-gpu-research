


/**
This kernel uses parallel reduction to find the 6 maximum and minimum points
in the point cloud
*/
__global__ findBoundingBoxKernel(GP_CLOUD_F4 &pc, int *minX, int *maxX,
                                int *minY, int *maxY, int *minZ, int *maxZ){
    //Would it be better to do multiple parallel reductions than one large memory consuming reduction?
    //This method makes 6 copies of the point cloud to find the necessary values 
    __shared__ int localMinX[MAX_THREADS/2];
    __shared__ int localMaxX[MAX_THREADS/2];
    __shared__ int localMinY[MAX_THREADS/2];
    __shared__ int localMaxY[MAX_THREADS/2];
    __shared__ int localMinZ[MAX_THREADS/2];
    __shared__ int localMaxZ[MAX_THREADS/2];
    __shared__ sl::float4 data[MAX_THREADS];
    int actualIndex = threadIdx.x+blockIdx.x*blockDim.x;
    bool notFull = false;

    if(actualIndex < pc.size){ //only write to shared memory if threads about to die
        data[threadIdx.x] = pc.data[actualIndex]; //Write from global memory into shared memory
    }
    else { //Accounts for final block with more threads than points
        notFull = true;
    }

    __syncthreads(); //At this point all data has been populated
    
    int aliveThreads = MAX_THREADS / 2;

    if(!notFull){ //Don't have to worry about checking for going out of bounds
    
        int minX = threadIdx.x, maxX = minX, minY = minX,
        maxY = minX, minZ = minX, maxZ = minZ; //initialize local indices of mins and maxes
        
        //Hard coding first iteration in order to save memory
        if (threadIdx.x < aliveThreads) {
            minX = (data[aliveThreads + threadIdx.x].x < data[minX].x)) ? aliveThreads + threadIdx.x : minX;
            maxX = (data[aliveThreads + threadIdx.x].x > data[maxX].x)) ? aliveThreads + threadIdx.x : maxX;
            minY = (data[aliveThreads + threadIdx.x].y < data[minY].y)) ? aliveThreads + threadIdx.x : minY;
            maxY = (data[aliveThreads + threadIdx.x].y > data[maxY].y)) ? aliveThreads + threadIdx.x : maxY;
            minZ = (data[aliveThreads + threadIdx.x].z < data[minZ].z)) ? aliveThreads + threadIdx.x : minZ;
            minX = (data[aliveThreads + threadIdx.x].z > data[maxZ].z)) ? aliveThreads + threadIdx.x : maxZ;
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

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x)) ? aliveThreads + threadIdx.x : minX;
                maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x)) ? aliveThreads + threadIdx.x : maxX;
                minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y)) ? aliveThreads + threadIdx.x : minY;
                maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y)) ? aliveThreads + threadIdx.x : maxY;
                minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z)) ? aliveThreads + threadIdx.x : minZ;
                minX = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z)) ? aliveThreads + threadIdx.x : maxZ;
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
        maxY = minX, minZ = minX, maxZ = minZ; //initialize local indices of mins and maxes checking for those exceeding size

        //Hard coding first iteration in order to save memory
        if (threadIdx.x < aliveThreads) {
            if(aliveThreads + threadIdx.x + blockDim.x*blockIdx.x < pc.size) { //If points to valid data
                minX = (data[aliveThreads + threadIdx.x].x < data[minX].x)) ? aliveThreads + threadIdx.x : minX;
                maxX = (data[aliveThreads + threadIdx.x].x > data[maxX].x)) ? aliveThreads + threadIdx.x : maxX;
                minY = (data[aliveThreads + threadIdx.x].y < data[minY].y)) ? aliveThreads + threadIdx.x : minY;
                maxY = (data[aliveThreads + threadIdx.x].y > data[maxY].y)) ? aliveThreads + threadIdx.x : maxY;
                minZ = (data[aliveThreads + threadIdx.x].z < data[minZ].z)) ? aliveThreads + threadIdx.x : minZ;
                minX = (data[aliveThreads + threadIdx.x].z > data[maxZ].z)) ? aliveThreads + threadIdx.x : maxZ;
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

        //Utilizes local arrays to keep track of values instead of hardcoded above
        while (aliveThreads > 0) {
            if (threadIdx.x < aliveThreads) {
                if(localMinX[aliveThreads + threadIdx.x] >= 0) { //If valid value compare and choose appropriately
                    minX = (data[localMinX[aliveThreads + threadIdx.x]].x < data[minX].x)) ? aliveThreads + threadIdx.x : minX;
                    maxX = (data[localMaxX[aliveThreads + threadIdx.x]].x > data[maxX].x)) ? aliveThreads + threadIdx.x : maxX;
                    minY = (data[localMinY[aliveThreads + threadIdx.x]].y < data[minY].y)) ? aliveThreads + threadIdx.x : minY;
                    maxY = (data[localMaxY[aliveThreads + threadIdx.x]].y > data[maxY].y)) ? aliveThreads + threadIdx.x : maxY;
                    minZ = (data[localMinZ[aliveThreads + threadIdx.x]].z < data[minZ].z)) ? aliveThreads + threadIdx.x : minZ;
                    minX = (data[localMaxZ[aliveThreads + threadIdx.x]].z > data[maxZ].z)) ? aliveThreads + threadIdx.x : maxZ;
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
    minX[blockIdx.x] = localMinX[0] + blockDim.x*blockIdx.x;
    maxX[blockIdx.x] = localMaxX[0] + blockDim.x*blockIdx.x;
    minY[blockIdx.x] = localMinY[0] + blockDim.x*blockIdx.x;
    maxY[blockIdx.x] = localMaxY[0] + blockDim.x*blockIdx.x;
    minZ[blockIdx.x] = localMinZ[0] + blockDim.x*blockIdx.x;
    maxZ[blockIdx.x] = localMaxZ[0] + blockDim.x*blockIdx.x;
}

/*
Finds the 6 maximum and minimum points needed to define a bounding box around the 
point cloud. Performs a function 6 times to find each point. The maximum pc size
for this function is 1048576 since it assumes the resulting reduction fits into a block
*/
void EuclideanClusterExtractor::findBoundingBox(GP_CLOUD_F4 &pc){
    int blocks = ceilDiv(pc.size,MAX_THREADS);
    int threads = MAX_THREADS;
    int minX[blocks]; //Stores max and min x,y,z values for each block in global memory
    int maxX[blocks];
    int minY[blocks]; 
    int maxY[blocks];
    int minZ[blocks]; 
    int maxZ[blocks];
    
    findBoundingBoxKernel<<<blocks,threads>>>(pc, minX, maxX, minY, maxY, minZ, maxZ); //Find 6 bounding values for all blocks
    
}
/*
This kernel determines the structure of the graph but does not build it
In theory, there is a memory-compute trade off to be made here. This kernel
is not strictly necessary if we allow an upper bound of memory so that each 
point can have the entire dataset amount of neighbors. Perhaps we can 
explore this allocation method instead.
*/
//b: enough, t: each point
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


/* This kernel builds the graph 
Fairly standard adjacency list structure. 
*/
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

EuclideanClusterExtractor::EuclideanClusterExtractor(float tolerance, int minSize, float maxSize) 
: tolerance(tolerance), minSize(minSize), maxSize(maxSize) {}

EuclideanClusterExtractor::extractClusters(GPU_Cloud_F4 pc) {

}