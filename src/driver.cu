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
    sl::Resolution cloud_res(320, 180);
    //sl::Resolution cloud_res_temp(10, 1);
    int k = 0;
    
    //Setup camera and viewer
    zed.open(); 
    GLViewer viewer;
    auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);

    //This is a cloud with data stored in GPU memory that can be acessed from CUDA kernels
    sl::Mat gpu_cloud (cloud_res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    int pcSize = cloud_res.area(); 
    cout << "Point clouds are of size: " << pcSize << endl;

    //This is a RANSAC model that we will use
    //RansacPlane ransac(sl::float3(0, 1, 0), 10, 400, 100, pcSize);

    //Temporary DEBUG ransac model:
    int testcloudsize = 10;
    RansacPlane testsac(sl::float3(0, 1, 0), 10, 5, 0.01, testcloudsize);
    GPU_Cloud_F4 testcloud;
    cudaMalloc(&testcloud.data , sizeof(sl::float4) * testcloudsize);
    testcloud.size = testcloudsize;
    sl::float4 dataCPU[testcloudsize] = {
        sl::float4(0.1, 0, 0, 4545), 
        sl::float4(10, 0, 0, 4545),
        sl::float4(-10, 0, 0.4, 4545),
        sl::float4(0, 0, 10, 4545),
        sl::float4(10, 0, 10, 4545),
        sl::float4(-10, 0, 10,4545),
        sl::float4(-5, 3, 10,4545),
        sl::float4(5, 2, 5,4545),
        sl::float4(2, 5, 2,4545),
        sl::float4(4, -4, 2,4545),
    };
    cudaMemcpy(testcloud.data, dataCPU, sizeof(sl::float4) * testcloudsize, cudaMemcpyHostToDevice);
    for(int i = 0; i < 1; i++) {
        RansacPlane::Plane planePoints = testsac.computeModel(testcloud);
        cout << "ransac says p1: " << planePoints.p1 << endl;
        cout << "ransac says p2: " << planePoints.p2 << endl;
        cout << "ransac says p3: " << planePoints.p3 << endl;
        cout << " ------------------------------------------- " << endl << endl;
    }
    
    while(true) {};
    return 1;

    auto start = high_resolution_clock::now(); 
    while(viewer.isAvailable()) {
        //Todo, Timer class. Timer.start(), Timer.record() 
        //grab the current point cloud
        k++;
        auto grabStart = high_resolution_clock::now();

        zed.grab();
        zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, cloud_res); 
        GPU_Cloud pc = getRawCloud(gpu_cloud);
        GPU_Cloud_F4 pc_f4 = getRawCloud(gpu_cloud, true);
        auto grabEnd = high_resolution_clock::now();
        auto grabDuration = duration_cast<microseconds>(grabEnd - grabStart); 
        cout << "grab time: " << (grabDuration.count()/1.0e3) << " ms" << endl; 
        
        /*
        auto ransacStart = high_resolution_clock::now();
        RansacPlane::Plane planePoints = ransac.computeModel(pc_f4);
        auto ransacStop = high_resolution_clock::now();
        auto ransacDuration = duration_cast<microseconds>(ransacStop - ransacStart); 
        cout << "ransac time: " << (ransacDuration.count()/1.0e3) << " ms" <<  endl; 
        */

        //run a custom CUDA filter that will color the cloud blue
        //TestFilter test_filter(gpu_cloud);
        //test_filter.run();
        /*
        auto blueStart = high_resolution_clock::now();
        TestFilter_F4 badTest(gpu_cloud);
        badTest.run();
        auto blueEnd = high_resolution_clock::now();
        auto blueDuration = duration_cast<microseconds>(blueEnd - blueStart); 
        cout << "blue time: " << (blueDuration.count()/1.0e3) << " ms" << endl;
        //Run RANSAC 
        */
        
        //update the viewer, the points will be blue
        //updateRansacPlane(sl::float3(-100, 0, 0), sl::float3(100, 0, 0), sl::float3(100, 100, 0), 1.5);



        //updateRansacPlane(planePoints.p1, planePoints.p2, planePoints.p3, 1.5);

        //viewer.updatePointCloud(testcloud);
        viewer.updatePointCloud(gpu_cloud);
    }
    gpu_cloud.free();
    zed.close();
    return 1;
}
