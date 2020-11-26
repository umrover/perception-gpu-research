#include <sl/Camera.hpp>
#include "GLViewer.hpp"
#include "test-filter.hpp"
#include "plane-ransac.hpp"
#include <iostream>
#include "common.hpp"

/*
Temporary driver program, do NOT copy this to mrover percep code at time of integration
Use/update existing Camera class which does the same thing but nicely abstracted.
*/


sl::Camera zed;

int main(int argc, char** argv) {
    //Setup camera and viewer
    zed.open(); 
    GLViewer viewer;
    auto camera_config = zed.getCameraInformation().camera_configuration;
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);

    //This is a cloud with data stored in GPU memory that can be acessed from CUDA kernels
    sl::Mat gpu_cloud (camera_config.resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

    while(viewer.isAvailable()) {
        //grab the current point cloud
        zed.grab();
        zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU); 
        
        GPU_Cloud pc = getRawCloud(gpu_cloud);

        //run a custom CUDA filter that will color the cloud blue
        //TestFilter test_filter(gpu_cloud);
        //test_filter.run();

        //Run RANSAC 
        
        
        //update the viewer, the points will be blue
        viewer.updatePointCloud(gpu_cloud);
        
    }
    gpu_cloud.free();
    zed.close();
    return 1;
}
