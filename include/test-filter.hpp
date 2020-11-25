#include <sl/Camera.hpp>
#include "common.hpp"


#ifndef TEST_FILTER
#define TEST_FILTER


class TestFilter {

    public:
        //Initialized filter with point cloud
        TestFilter(sl::Mat gpu_cloud);
        
        //Run the filter
        void run();

    private:
        GPU_Cloud gpu_cloud;

};

#endif