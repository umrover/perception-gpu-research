#include <sl/Camera.hpp>
#include "GLViewer.hpp"



//void ZedToPcl();
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