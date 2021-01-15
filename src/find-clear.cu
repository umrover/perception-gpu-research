#include "common.hpp"
#include <float.h>
//Find clear path
//Checks to see if minx and maxX are between the two projected lines
//It not checks to see if the line segment between the two minX and maxX intersect the projected line
//If such points are found launch write them to place in shared memory
//If such poitns are not found then in shared memory space write float_max or float_min
//SyncThreads and perform parallel reduction to find minX and maxX thread values
//launch left and right path kernels using new calculated lines
//Left and right path kernels will perform the same steps except in a particular direction
//Assumes we don't have over 1024 clusters

class compareLine {
public:
    int xIntercept;
    double slope;
    
    __device__ compareLine(double angle_in, int xInt_in) : xIntercept{xInt_in}, 
                        slope{tan(angle_in*PI/180)} {
                            if(slope != 0) {
                                slope = 1/slope;
                            }
                        }

    //Returns 1 if point is right of line, 0 if on, -1 if left of line
    __device__ int operator()(float x, float y) {
        
        //Make sure don't divide by 0
        double xc = xIntercept; //x calculated
        if(slope != 0) {
            xc = y/slope+xIntercept; //Find x value on line with same y value as input point
        }
            
        //Point is right of the line
        if(x > xc) {
            return 1;
        }
        //Point is on the line
        else if (x == xc) {
            return 0;
        }
        //Point is left of the line
        else {
            return -1;
        } 
    }

    //Assumes x1 < x2
    __device__ bool operator()(float x1, float y1, float x2, float y2) {
        if(x1 != x2){
            if(slope != 0){
                float slopeSeg = (y2-y1)/(x2-x1);
                float xIntersect = (-slopeSeg*x1+y1-xIntercept)/(slope-slopeSeg);
                return (xIntersect < x2 && xIntersect > x1);
            }
            //Check if left of line and right of line if slope is undefined
            else if(this->operator()(x1,y1) < 1 && this->operator()(x2,y2) > -1) return true; 
        }
        return false;
        
    }
};

int findNextLargestSquare(int num){
    int exp = log2(num) + 1;
    return pow(2,exp);
}

__global__ void findClearPathKernel(float* minXG, float* maxXG, float* minZG, int numClusters) {
        
    //Declare variables
    const int HALF_ROVER = 5;
    if(threadIdx.x >= numClusters) return;

    __shared__ float maxXS[MAX_THREADS];
    __shared__ float minXS[MAX_THREADS];
    __shared__ float minZS[MAX_THREADS];
    compareLine rightLine(0,HALF_ROVER); //Is there a way to make these objects shared?
    compareLine leftLine(0,-HALF_ROVER);

    float maxX;
    float minX;
    float minZ;

    //Copy over data from global to local
    maxX = maxXG[threadIdx.x];
    minX = minXG[threadIdx.x];
    minZ = minZG[threadIdx.x];

    if(threadIdx.x == 0)
        printf("Vars are copied\n");
    //Check where point is relative to line
    //Since the z values of the min and max x values aren't stored we are going
    //to assume that they are both located at the min z value of the cluster

    //Checks if either of the points is between the two lines
    if((leftLine(maxX, minZ) > -1 && rightLine(maxX, minZ) < 1) || //check if maxX is between right and left
       (leftLine(minX, minZ) > -1 && rightLine(minX, minZ) < 1) || //Check if minX is between right and left
       (leftLine(minX, minZ, maxX, minZ) || rightLine(minZ, minZ, maxX, minZ))) { //check if lines intersect line seg
        maxXS[threadIdx.x] = maxX;
        minXS[threadIdx.x] = minX;
        minZS[threadIdx.x] = minZ;
        printf("Point %d found in path\n", threadIdx.x);
    }
    else {
        maxXS[threadIdx.x] = FLT_MIN;
        minXS[threadIdx.x] = FLT_MAX;
    }

    __syncthreads();

    //Iterate through points to find mins and maxes
    //No need for parallel reduction, since cluster size is relatively small
    
    if(threadIdx.x == 0){
        float minXFinal = FLT_MAX, minZMinXFinal = 0, maxXFinal = FLT_MIN, minZMaxXFinal = 0; 
        for(int i = 0; i < numClusters; ++i ){
            if(maxXS[i] > maxXFinal) {
                maxXFinal= maxXS[i];
                minZMaxXFinal = minZS[i];
                printf("new max found\n");
            }
            if(minXS[i] < minXFinal) {
                minXFinal = minXS[i];
                minZMinXFinal = minZS[i];
                printf("new min found\n");
            }
        }
        float leftBearing;
        leftBearing = (maxXFinal != FLT_MIN) ? 100 : 0; //There is an obstacle in the path

        printf("Max: (%.1f, %.1f)\n", maxXFinal, minZMaxXFinal);
        printf("Min: (%.1f, %.1f)\n", minXFinal, minZMinXFinal);
    }
    
}

void testClearPath() {

    int testClusterSize = 10;
    float* minXG;
    float* maxXG;
    float* minZG;

    cudaMalloc(&minXG, sizeof(float)*testClusterSize);
    cudaMalloc(&maxXG, sizeof(float)*testClusterSize);
    cudaMalloc(&minZG, sizeof(float)*testClusterSize);

    float minXCPU[testClusterSize] = { -6,   -6, 10, 10, 10,   -10, 10, 10, 10, 10};
    float maxXCPU[testClusterSize] = { -6, -5.5, 20, 20, 20,  -9.5, 20, 20, 20, 20};
    float minZCPU[testClusterSize] = { 10,   10, 10, 10, 10,    40, 10, 10, 10, 10};

    cudaMemcpy(minXG, minXCPU, sizeof(float)*testClusterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(maxXG, maxXCPU, sizeof(float)*testClusterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(minZG, minZCPU, sizeof(float)*testClusterSize, cudaMemcpyHostToDevice);

    findClearPathKernel<<<1, MAX_THREADS>>>(minXG, maxXG, minZG, testClusterSize);
    checkStatus(cudaGetLastError());
    cudaDeviceSynchronize();

    cudaFree(minXG);
    cudaFree(maxXG);
    cudaFree(minZG);
}