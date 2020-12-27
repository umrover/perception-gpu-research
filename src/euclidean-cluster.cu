#include "euclidean-cluster.hpp"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
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
    //printf("%d: %d \n",ptIdx, neighborCount );
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
    if(ptIdx >= pc.size) return;

    //debug lines
   // if(threadIdx.x == 0) *m = false;
   // __syncthreads();
   // printf("pt idx: %d, label: %d, flag: %d frontier one: %d frontier two: %d \n", ptIdx, labels[ptIdx], (*m) ? 1 : 0, f1[ptIdx] ? 1 : 0, f2[ptIdx] ? 1 : 0);

    bool mod = false;
    //TODO, load the NEIGHBOR list to shared memory 
    if(f1[ptIdx]) {
        //printf("active frontier %d \n", ptIdx);

        int* list = neighborLists + listStart[ptIdx];
        int listLen = listStart[ptIdx+1] - listStart[ptIdx];
        f1[ptIdx] = false;
        int myLabel = labels[ptIdx];

        //printf("[len] pt idx: %d, list-len: %d \n", ptIdx, listLen);

        for(int i = 0; i < listLen; i++) {
            int otherLabel = labels[list[i]];
            if(myLabel < otherLabel) { //are these reads actually safe?
                //printf("-- updating other: %d to be %d \n", otherLabel, myLabel);

                atomicMin(&labels[list[i]], myLabel);
                f2[list[i]] = true;
                *m = true;
            } else if(myLabel > otherLabel) {
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

    /*
    __syncthreads();
    if(threadIdx.x == 0) {
    if(*m) printf("still going \n");
    else printf("done \n");
    }*/
}

//this debug kernel colors points based on their label
__global__ void colorClusters(GPU_Cloud_F4 pc, int* labels, int* keys, int* values) {
    int ptIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(ptIdx >= pc.size) return;

    int i = 0;
    while(true) {
        if(labels[ptIdx] == keys[i]) {
            if(values[i] < 60) return;
            else break;
        }
        i++;
    }
    
    float red = 3.57331108403e-43;
    float green = 9.14767637511e-41;
    float blue = 2.34180515203e-38;
    float magenta = 2.34184088514e-38; 
    float yellow = 9.18340948595e-41;
    
    int lbl = labels[ptIdx] % 5;
    
    if(lbl == 0) pc.data[ptIdx].w = red;
    if(lbl == 1) pc.data[ptIdx].w = green;
    if(lbl == 2) pc.data[ptIdx].w = blue;
    if(lbl == 3) pc.data[ptIdx].w = magenta;
    if(lbl == 4) pc.data[ptIdx].w = yellow; 

}

EuclideanClusterExtractor::EuclideanClusterExtractor(float tolerance, int minSize, float maxSize, GPU_Cloud_F4 pc) 
: tolerance(tolerance), minSize(minSize), maxSize(maxSize) {

    cudaMalloc(&listStart, sizeof(int)*(pc.size+1));
    cudaMalloc(&labels, sizeof(int)*pc.size);
    cudaMalloc(&f1, sizeof(bool)*pc.size);
    cudaMalloc(&f2, sizeof(bool)*pc.size);
    cudaMalloc(&stillGoing, sizeof(bool));

   // colorClusters<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, nullptr);
}

//perhaps use dynamic parallelism 
void EuclideanClusterExtractor::extractClusters(GPU_Cloud_F4 pc) {
    if(pc.size == 0) return;
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
    //std::cout << "total adj size: " << totalAdjanecyListsSize << std::endl;
    
    cudaMalloc(&neighborLists, sizeof(int)*totalAdjanecyListsSize);
    buildGraphKernel<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, tolerance, neighborLists, listStart, labels, f1, f2);
    checkStatus(cudaDeviceSynchronize());

    //int* temp2 = (int*) malloc(sizeof(int)*(totalAdjanecyListsSize));
    //checkStatus(cudaMemcpy(temp2, neighborLists, sizeof(int)*(totalAdjanecyListsSize), cudaMemcpyDeviceToHost));
    //for(int i = 0; i < totalAdjanecyListsSize; i++) std::cout << "neighbor list: " << temp2[i] << std::endl;
    
    /*
    for(int i = 0; i < 4; i++) {
    bool flag = false;
    cudaMemcpy(stillGoing, &flag, sizeof(bool), cudaMemcpyHostToDevice);
    propogateLabels<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, neighborLists, listStart, labels, f1, f2, stillGoing);
    bool* t = f1;
    f1 = f2;
    f2 = t;
    }*/

    
    bool stillGoingCPU = true;    
    while(stillGoingCPU) {
        //one iteration of label propogation
        stillGoingCPU = false;
        cudaMemcpy(stillGoing, &stillGoingCPU, sizeof(bool), cudaMemcpyHostToDevice);
        propogateLabels<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, neighborLists, listStart, labels, f1, f2, stillGoing);

        //swap the frontiers
        bool* t = f1;
        f1 = f2;
        f2 = t;

        //get flag to see if we are done
        cudaMemcpy(&stillGoingCPU, stillGoing, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    
    thrust::device_vector<int> labelsSorted(pc.size);
    thrust::device_vector<int> count(pc.size, 1);
    thrust::device_vector<int> keys(pc.size);
    thrust::device_vector<int> values(pc.size);


    thrust::copy(thrust::device, labels, labels+pc.size, labelsSorted.begin());
    thrust::sort(thrust::device, labelsSorted.begin(), labelsSorted.end());
    thrust::reduce_by_key(thrust::device, labelsSorted.begin(), labelsSorted.end(), count.begin(), keys.begin(), values.begin());
    int* gpuKeys = thrust::raw_pointer_cast( keys.data() );
    int* gpuVals = thrust::raw_pointer_cast( values.data() );


    colorClusters<<<ceilDiv(pc.size, MAX_THREADS), MAX_THREADS>>>(pc, labels, gpuKeys, gpuVals);
    checkStatus(cudaDeviceSynchronize()); //not needed?
    cudaFree(neighborLists);
}

EuclideanClusterExtractor::~EuclideanClusterExtractor() {

}