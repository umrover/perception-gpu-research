
#include <sl/Camera.hpp>
#include "GLViewer.hpp"
//#include "test-filter.hpp"
#include "plane-ransac.hpp"
#include <iostream>
#include "common.hpp"
#include <Eigen/Dense>
#include <chrono> 
#include "test-filter.hpp"
#include "pcl.hpp"
#include<unistd.h>
#include <thread>
#include "pass-through.hpp"
#include "euclidean-cluster.hpp"
#include "driver.hpp"

using namespace std::chrono; 

/*
Temporary driver program, do NOT copy this to mrover percep code at time of integration
Use/update existing Camera class which does the same thing but nicely abstracted.
*/

//#define USE_PCL

//Zed camera and viewer
sl::Camera zed;
GLViewer viewer;
int guiK = 0;

RansacPlane::Plane planePoints;
EuclideanClusterExtractor::ObsReturn obstacles;

//This is a thread to just spin the zed viewer
void spinZedViewer() {
    while(viewer.isAvailable()) {
        std::this_thread::sleep_for (std::chrono::milliseconds(10));
     //   updateRansacPlane(planePoints.p1, planePoints.p2, planePoints.p3, 600.5);
     
    /*   float minX[] = {0.0};
        float maxX[] = {100.0};
        float minY[] = {0.0};
        float maxY[] = {100.0};
        float minZ[] = {0.0};
        float maxZ[] = {100.0}; */
        updateObjectBoxes(obstacles.size, obstacles.minX, obstacles.maxX, obstacles.minY, obstacles.maxY, obstacles.minZ, obstacles.maxZ );
      // updateObjectBoxes(1, minX, maxX, minY, maxY, minZ, maxZ );
    }
}

void nextFrame() {
    guiK++;
    cout << guiK << endl;
}
void prevFrame() {
    guiK--;
    cout << guiK << endl;
}

int main(int argc, char** argv) {  

    testClearPath();
    
    return 1;
}
