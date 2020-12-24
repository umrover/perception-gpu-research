#include "euclidean-cluster.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include "common.hpp"

/*
This kernel determines the structure of the graph but does not build it
In theory, there is a memory-compute trade off to be made here. This kernel
is not strictly necessary if we allow an upper bound of memory so that each 
point can have the entire dataset amount of neighbors. Perhaps we can 
explore this allocation method instead.
*/
//b: enough, t: each point
__global__ void determineGraphStructureKernel(GPU_Cloud_F4 pc, float tolerance, int* listStart) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    
    //horrible slow way of doing this that is TEMPORARY --> please switch to radix sorted bins
    for(int i = 0; i < pc.size; i++) {
        sl::float3 dvec = (pt - sl::float3(pc.data[i]));
        //this is a neighbor
        if( dvec.norm() < tolerance && i != ptIdx) {
            neighborCount++;
        }
    }
    listStart[ptIdx] = neighborCount;

    //we must do an exclusive scan using thrust after this kernel
    printf("%d: %d \n",ptIdx, neighborCount );
}


/* This kernel builds the graph 
Fairly standard adjacency list structure. 
*/
__global__ void buildGraphKernel(GPU_Cloud_F4 pc, float tolerance, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    sl::float3 pt = pc.data[ptIdx];
    int neighborCount = 0;
    //get the adjacency list for this point
    int* list = neighborLists + listStart[ptIdx];
    
    //horrible slow way of doing this that is TEMPORARY --> please switch to radix sorted bins
    for(int i = 0; i < pc.size; i++) {

        sl::float3 dvec = (pt - sl::float3(pc.data[i]));
        //this is a neighbor
        if( dvec.norm() < tolerance && i != ptIdx) {
            list[neighborCount] = i;
            neighborCount++;
        }
    }
    
    labels[ptIdx] = ptIdx;
    f1[ptIdx] = true;
    f2[ptIdx] = false;
}

/*
this kernel propogates labels, it must be called in a loop until its flag "m" is false, indicating
no more changes are pending. 
*/
//each thread is a point 
__global__ void propogateLabels(GPU_Cloud_F4 pc, int* neighborLists, int* listStart, int* labels, bool* f1, bool* f2, bool* m) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    bool mod = false;
    //TODO, load the NEIGHBOR list to shared memory 
    if(f1[ptIdx]) {
        int* list = neighborLists + listStart[ptIdx];
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

        if(mod) {
            atomicMin(&labels[ptIdx], myLabel);
            f2[ptIdx] = true;
            *m = true;
        }
    } 


}

//this debug kernel colors points based on their label
__global__ void colorClusters(GPU_Cloud_F4 pc, int* labels) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;
    printf("dat %d: %f, %f, %f \n", ptIdx,  pc.data[ptIdx].x, pc.data[ptIdx].y,  pc.data[ptIdx].z );
}

EuclideanClusterExtractor::EuclideanClusterExtractor(float tolerance, int minSize, float maxSize, GPU_Cloud_F4 pc) 
: tolerance(tolerance), minSize(minSize), maxSize(maxSize) {

    cudaMalloc(&listStart, sizeof(int)*(pc.size+1));
    cudaMalloc(&labels, sizeof(int)*pc.size);
    cudaMalloc(&f1, sizeof(bool)*pc.size);
    cudaMalloc(&f2, sizeof(bool)*pc.size);
    cudaMalloc(&stillGoing, sizeof(bool));

    colorClusters<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, nullptr);
}

//perhaps use dynamic parallelism 
void EuclideanClusterExtractor::extractClusters(GPU_Cloud_F4 pc) {
    //set frontier arrays appropriately [done in build graph]
    //checkStatus(cudaMemsetAsync(f1, 1, sizeof(pc.size)));
    //checkStatus(cudaMemsetAsync(f2, 0, sizeof(pc.size)));
    determineGraphStructureKernel<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, listStart);
    thrust::exclusive_scan(thrust::device, listStart, listStart+pc.size+1, listStart, 0);
    checkStatus(cudaDeviceSynchronize());
    int totalAdjanecyListsSize;
    /*//debugint* temp = (int*) malloc(sizeof(int)*(pc.size+1));
    checkStatus(cudaMemcpy(temp, listStart, sizeof(int)*(pc.size+1), cudaMemcpyDeviceToHost));
    for(int i = 0; i < pc.size+1; i++) std::cout << "ex scan: " << temp[i] << std::endl; */
    checkStatus(cudaMemcpy(&totalAdjanecyListsSize, &listStart[pc.size], sizeof(int), cudaMemcpyDeviceToHost));
    std::cout << "total adj size: " << totalAdjanecyListsSize << std::endl;
    
    cudaMalloc(&neighborLists, sizeof(int)*totalAdjanecyListsSize);
    buildGraphKernel<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, neighborLists, listStart, labels, f1, f2);
    checkStatus(cudaDeviceSynchronize());

    int* temp2 = (int*) malloc(sizeof(int)*(totalAdjanecyListsSize));
    checkStatus(cudaMemcpy(temp2, neighborLists, sizeof(int)*(totalAdjanecyListsSize), cudaMemcpyDeviceToHost));
    for(int i = 0; i < totalAdjanecyListsSize; i++) std::cout << "neighbor list: " << temp2[i] << std::endl;
    
    /*
    int* totalAdjanecyListsSize;
    checkStatus(cudaMemcpy(&totalAdjanecyListsSize, &listStart[pc.size-1], sizeof(int), cudaMemcpyDeviceToHost));
    
    bool stillGoingCPU = true;    
    while(stillGoingCPU) {
    
        propogateLabels<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, neighborLists, listStart, labels, f1, f2, stillGoing);

        //swap the frontiers
        bool* t = f1;
        f1 = f2;
        f2 = t;
        //get flag to see if we are done
        cudaMemcpy(&stillGoingCPU, stillGoing, sizeof(bool), cudaMemcpyDeviceToHost);
    }*/
}

EuclideanClusterExtractor::~EuclideanClusterExtractor() {

}