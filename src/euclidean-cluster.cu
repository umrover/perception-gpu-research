



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