#include <sl/Camera.hpp>


//void ZedToPcl();
//void PclToZed();

sl::Camera zed;

void grabPointCloud() {
    sl::Mat data_cloud;
    zed.retrieveMeasure(data_cloud, sl::MEASURE::XYZRGBA, sl::MEM::CPU);


    //Populate Point Cloud
    float *p_data_cloud = data_cloud.getPtr<float>();
}


int main() {
    zed.open(); //== sl::ERROR_CODE::SUCCESS);

    while(true) {

    }
    return 1;
}