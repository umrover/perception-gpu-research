
#include <sl/Camera.hpp>
#include "GLViewer.hpp"
//#include "test-filter.hpp"
#include "plane-ransac.hpp"
#include <iostream>
#include "common.hpp"
#include <Eigen/Dense>
#include <chrono> 
#include "test-filter-f4.hpp"

#include "pcl.hpp"

using namespace std::chrono; 

/*
Temporary driver program, do NOT copy this to mrover percep code at time of integration
Use/update existing Camera class which does the same thing but nicely abstracted.
*/


sl::Camera zed;

int main(int argc, char** argv) {  
    
    sl::Resolution cloud_res(320/2, 180/2);
    //sl::Resolution cloud_res_temp(10, 1);
    int k = 0;
    
    //Setup camera and viewer
    sl::InitParameters init_params;
    init_params.coordinate_units = sl::UNIT::MILLIMETER;
    zed.open(init_params); 
    GLViewer viewer;
    auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);

    //This is a cloud with data stored in GPU memory that can be acessed from CUDA kernels
    sl::Mat gpu_cloud (cloud_res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    int pcSize = cloud_res.area(); 
    cout << "Point clouds are of size: " << pcSize << endl;

    //This is a RANSAC model that we will use
    RansacPlane ransac(sl::float3(0, 1, 0), 10, 400, 100.8, pcSize);
    

    auto start = high_resolution_clock::now(); 
    
    
    int iter = 0;
	readData(); //Load the pcd file names into pcd_names
	setPointCloud(120); //Set the first point cloud to be the first of the files
    pclViewer = createRGBVisualizer(pc_pcl);
    /*
	pclViewer = createRGBVisualizer(pc); //Create an RGB visualizer for said cloud
    while(true) {

		iter++; 
		pclViewer->updatePointCloud(pc); //update the viewer 
    	pclViewer->spinOnce(10); //pcl spin wait
		setPointCloud(iter); //load next point cloud into the same pointer
	}*/

    while(viewer.isAvailable()) {
        //Todo, Timer class. Timer.start(), Timer.record() 
        //grab the current point cloud
        k++;

       // if(k > 500) setPointCloud(k-500);
        //else continue;
        //sl::Mat pclTest(sl::Resolution(320, 180), sl::MAT_TYPE::F32_C4, sl::MEM::CPU);
        //pclToZed(pclTest, pc_pcl);

        auto grabStart = high_resolution_clock::now();

        /*
       //zed.grab();
        //zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, cloud_res); 
        GPU_Cloud pc = getRawCloud(gpu_cloud);
        GPU_Cloud_F4 pc_f4 = getRawCloud(gpu_cloud, true);
        auto grabEnd = high_resolution_clock::now();
        auto grabDuration = duration_cast<microseconds>(grabEnd - grabStart); 
        cout << "grab time: " << (grabDuration.count()/1.0e3) << " ms" << endl; 
        
        
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
        
        zed.grab();
        zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, cloud_res); 
        ZedToPcl(pc_pcl, gpu_cloud);

        pclViewer->updatePointCloud(pc_pcl); //update the viewer 
    	pclViewer->spinOnce(10);
        
        //update the viewer, the points will be blue
        //updateRansacPlane(planePoints.p1, planePoints.p2, planePoints.p3, 600.5);

        //viewer.updatePointCloud(pclTest);
       // viewer.updatePointCloud(gpu_cloud);
    }
    gpu_cloud.free();
    zed.close(); 
    return 1;
}
