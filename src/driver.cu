#include <sl/Camera.hpp>
#include "GLViewer.hpp"
#include <pcl/common/common_headers.h>


/*
Temporary driver program, do NOT  copy to mrover percep code
Use/update existing Camera class which does the same thing 
*/


//This function convert a RGBA color packed into a packed RGBA PCL compatible format
/*
inline float convertColor(float colorIn) {
    uint32_t color_uint = *(uint32_t *) & colorIn;
    unsigned char *color_uchar = (unsigned char *) &color_uint;
    color_uint = ((uint32_t) color_uchar[0] << 16 | (uint32_t) color_uchar[1] << 8 | (uint32_t) color_uchar[2]);
    return *reinterpret_cast<float *> (&color_uint);
}

void ZedToPcl(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & p_pcl_point_cloud, sl::Mat zed_cloud) {
  sl::Mat zed_cloud_cpu;
  zed_cloud.copyTo(zed_cloud_cpu,  sl::COPY_TYPE::GPU_CPU);
 
  float* p_data_cloud = zed_cloud_cpu.getPtr<float>();
  int index = 0;
  for (auto &it : p_pcl_point_cloud->points) {
    float X = p_data_cloud[index];
    if (!isValidMeasure(X)) // Checking if it's a valid point
        it.x = it.y = it.z = it.rgb = 0;
    else {
        it.x = X;
        it.y = p_data_cloud[index + 1];
        it.z = p_data_cloud[index + 2];
        it.rgb = convertColor(p_data_cloud[index + 3]); // Convert a 32bits float into a pcl .rgb format
    }
    index += 4;
  }

} */
//void PclToZed();

sl::Camera zed;

void grabPointCloud() {
    sl::Mat data_cloud;
    zed.retrieveMeasure(data_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU);


    //Populate Point Cloud
    //float *p_data_cloud = data_cloud.getPtr<float>();
}


int main(int argc, char** argv) {
    zed.open(); //== sl::ERROR_CODE::SUCCESS);
    GLViewer viewer;
    auto camera_config = zed.getCameraInformation().camera_configuration;

    
    GLenum errgl = viewer.init(argc, argv, camera_config.calibration_parameters.left_cam);

    sl::Mat data_cloud (camera_config.resolution, sl::MAT_TYPE::F32_C4, sl::MEM::GPU);

    while(viewer.isAvailable()) {
        //sl::Mat data_cloud;
        zed.grab();
        zed.retrieveMeasure(data_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU);
        viewer.updatePointCloud(data_cloud);

    }
    data_cloud.free();
    zed.close();
    return 1;
}