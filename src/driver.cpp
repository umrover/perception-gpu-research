
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


#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


using namespace std::chrono; 

/*
Temporary driver program, do NOT copy this to mrover percep code at time of integration
Use/update existing Camera class which does the same thing but nicely abstracted.
*/

//#define USE_PCL

//Zed camera and viewer
sl::Camera zed;
GLViewer viewer;

//This is a thread to just spin the zed viewer
void spinZedViewer() {
    while(viewer.isAvailable()) {
        std::this_thread::sleep_for (std::chrono::milliseconds(10));
    }
}

int main(int argc, char** argv) {  
    /*
    sl::Resolution cloud_res(320/2, 180/2);
    int k = 0;
    
    //Setup camera and viewer
    sl::InitParameters init_params;
    init_params.coordinate_units = sl::UNIT::MILLIMETER;
    zed.open(init_params); 
    // init viewer
    auto camera_config = zed.getCameraInformation(cloud_res).camera_configuration;
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);

    //This is a cloud with data stored in GPU memory that can be acessed from CUDA kernels
    sl::Mat gpu_cloud (cloud_res, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);
    int pcSize = cloud_res.area(); 
    cout << "Point clouds are of size: " << pcSize << endl;

    //Pass Through Filter
    PassThrough passZ('z', 200.0, 7000.0); //This probly won't do much since range so large
   // PassThrough passY('y', 100.0, 600.0); 
    
    //This is a RANSAC model that we will use
    //sl::float3 axis, float epsilon, int iterations, float threshold,  int pcSize
    RansacPlane ransac(sl::float3(0, 1, 0), 7, 400, 150, pcSize); //change the threshold to 150
        
    //PCL integration variables
    int iter = 0;
	readData(); //Load the pcd file names into pcd_names
	setPointCloud(5); //Set the first point cloud to be the first of the files
    pclViewer = createRGBVisualizer(pc_pcl);

    #ifdef USE_PCL
    //slows down performance of all GPU functions since running in parallel with them
    thread zedViewerThread(spinZedViewer);
    #endif
    */

    //Temporary DEBUG model:
    
    //Create a point synthetic point cloud
    int testcloudsize = 10;
    GPU_Cloud_F4 testcloud;
    cudaMalloc(&testcloud.data , sizeof(sl::float4) * testcloudsize);
    testcloud.size = testcloudsize;
    sl::float4 dataCPU[testcloudsize] = {
        sl::float4(0, 0, 1, 4545), 
        sl::float4(0, 0, -15, 4545),
        sl::float4(-15, 0, 0, 4545),
        sl::float4(15, -15, 6, 4545),
        sl::float4(1, 1, -1, 4545),
        sl::float4(-15, 15, 15, 4545),
        sl::float4(-15, -15, 15, 4545),
        sl::float4(-15, 15, -15, 4545),
        sl::float4(15, 15, -15, 4545),
        sl::float4(-15, -15, 15, 4545),
    };
    cudaMemcpy(testcloud.data, dataCPU, sizeof(sl::float4) * testcloudsize, cudaMemcpyHostToDevice);
    
    EuclideanClusterExtractor ece(5, 0, 0, testcloud, 2);
    ece.findBoundingBox(testcloud);
    ece.buildBins(testcloud);
    ece.extractClusters(testcloud);
    ece.freeBins();
    /*
    GPU_Cloud_F4 tmp;
    tmp.size = cloud_res.width*cloud_res.height;
    EuclideanClusterExtractor ece(520, 50, 0, tmp, 4); //60/120

    while(true) {
        //Todo, Timer class. Timer.start(), Timer.record() 
        k++;

        //Grab cloud from PCD file
        #ifdef USE_PCL 
        setPointCloud(k);
        sl::Mat pclTest(sl::Resolution(320/2, 180/2), sl::MAT_TYPE::F32_C4, sl::MEM::CPU);
        pclToZed(pclTest, pc_pcl);
        GPU_Cloud_F4 pc_f4 = getRawCloud(pclTest, true);
        #endif
        //Grab cloud from the Zed camera
        
        #ifndef USE_PCL
        
        auto grabStart = high_resolution_clock::now();
        zed.grab();
        zed.retrieveMeasure(gpu_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, cloud_res); 
        GPU_Cloud pc = getRawCloud(gpu_cloud);
        GPU_Cloud_F4 pc_f4 = getRawCloud(gpu_cloud, true);
        auto grabEnd = high_resolution_clock::now();
        auto grabDuration = duration_cast<microseconds>(grabEnd - grabStart); 
        cout << "grab time: " << (grabDuration.count()/1.0e3) << " ms" << endl; 
        #endif
        
        //Run PassThrough Filter
        
        cout << "[size] pre-pass-thru: " << pc_f4.size << endl;
        
        auto passThroughStart = high_resolution_clock::now();
        passZ.run(pc_f4);
        //passY.run(pc_f4);
        auto passThroughStop = high_resolution_clock::now();
        auto passThroughDuration = duration_cast<microseconds>(passThroughStop - passThroughStart); 
        cout << "pass-through time: " << (passThroughDuration.count()/1.0e3) << " ms" <<  endl; 
        
       
        cout << "[size] pre-ransac: " << pc_f4.size << endl;
        clearStale(pc_f4, 320/2*180/2);
        //Perform RANSAC Plane segmentation to find the ground
        auto ransacStart = high_resolution_clock::now();
        RansacPlane::Plane planePoints = ransac.computeModel(pc_f4, true);
        auto ransacStop = high_resolution_clock::now();
        auto ransacDuration = duration_cast<microseconds>(ransacStop - ransacStart); 
        cout << "ransac time: " << (ransacDuration.count()/1.0e3) << " ms" <<  endl; 
        cout << "[size] post-ransac: " << pc_f4.size << endl; 
        clearStale(pc_f4, 320/2*180/2);

        //TestFilter tf;
        //tf.run(pc_f4); 

        
        auto eceStart = high_resolution_clock::now();
        ece.findBoundingBox(pc_f4);
        ece.buildBins(pc_f4);
        ece.extractClusters(pc_f4);
        ece.freeBins();
        auto eceStop = high_resolution_clock::now();
        auto eceDuration = duration_cast<microseconds>(eceStop - eceStart); 
        cout << "ECE time: " << (eceDuration.count()/1.0e3) << " ms" <<  endl; 
        
        #ifndef USE_PCL
        viewer.isAvailable();
        #endif

        //PCL viewer
        

        //PCL viewer + Zed SDK Viewer
        #ifdef USE_PCL
        ZedToPcl(pc_pcl, pclTest);
        pclViewer->updatePointCloud(pc_pcl); //update the viewer 
    	pclViewer->spinOnce(10);
        unsigned int microsecond = 1000000;
       // usleep(microsecond);
        viewer.updatePointCloud(pclTest);
        #endif


        //ZED sdk custom viewer ONLY
        #ifndef USE_PCL
        //draw an actual plane on the viewer where the ground is
        //updateRansacPlane(planePoints.p1, planePoints.p2, planePoints.p3, 600.5);
        viewer.updatePointCloud(gpu_cloud);
       // viewer.updatePointCloud(testcloudmat);

        #endif

        cerr << "Camera frame rate: " << zed.getCurrentFPS() << "\n";

        std::this_thread::sleep_for(0.2s);
    }
    gpu_cloud.free();
    zed.close(); 
    return 1;
    */
}
