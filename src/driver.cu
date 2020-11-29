#include <sl/Camera.hpp>
#include "GLViewer.hpp"
//#include "test-filter.hpp"
#include "plane-ransac.hpp"
#include <iostream>
#include "common.hpp"
#include <Eigen/Dense>
#include <chrono> 
#include "test-filter-f4.hpp"


using namespace std::chrono; 

/*
Temporary driver program, do NOT copy this to mrover percep code at time of integration
Use/update existing Camera class which does the same thing but nicely abstracted.
*/


sl::Camera zed;

int main(int argc, char** argv) {
    float4 f = make_float4(1, 2, 3, 4);

    //Setup camera and viewer
    zed.open(); 
    GLViewer viewer;
    auto camera_config = zed.getCameraInformation().camera_configuration;
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);

    //This is a cloud with data stored in GPU memory that can be acessed from CUDA kernels
    sl::Mat gpu_cloud (camera_config.resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

    int pcSize = zed.getCameraInformation().camera_resolution.area();
    cout << "Point clouds are of size: " << pcSize << endl;

    //This is a RANSAC model that we will use
    RansacPlane ransac(Eigen::Vector3d(0, 1, 0), 10, 400, 100, pcSize);

    auto start = high_resolution_clock::now(); 
    while(viewer.isAvailable()) {
        //Todo, Timer class. Timer.start(), Timer.record() 
        //grab the current point cloud
        auto grabStart = high_resolution_clock::now();
        zed.grab();
        zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU); 
        GPU_Cloud pc = getRawCloud(gpu_cloud);
        GPU_Cloud_F4 pc_f4 = getRawCloud(gpu_cloud, true);
        auto grabEnd = high_resolution_clock::now();
        auto grabDuration = duration_cast<microseconds>(grabEnd - grabStart); 
        cout << "grab time: " << (grabDuration.count()/1.0e3) << " ms" << endl; 
        
        auto ransacStart = high_resolution_clock::now();
        ransac.computeModel(pc_f4);
        auto ransacStop = high_resolution_clock::now();
        auto ransacDuration = duration_cast<microseconds>(ransacStop - ransacStart); 
        cout << "ransac time: " << (ransacDuration.count()/1.0e3) << " ms" <<  endl; 
        

        //run a custom CUDA filter that will color the cloud blue
        //TestFilter test_filter(gpu_cloud);
        //test_filter.run();

        auto blueStart = high_resolution_clock::now();
        TestFilter_F4 badTest(gpu_cloud);
        badTest.run();
        auto blueEnd = high_resolution_clock::now();
        auto blueDuration = duration_cast<microseconds>(blueEnd - blueStart); 
        cout << "blue time: " << (blueDuration.count()/1.0e3) << " ms" << endl;
        //Run RANSAC 
        
        
        //update the viewer, the points will be blue
        viewer.updatePointCloud(gpu_cloud);
    }
    gpu_cloud.free();
    zed.close();
    return 1;
}
