#pragma once
#include <sl/Camera.hpp>
#include "common.hpp"


#ifndef PASS_THROUGH
#define PASS_THROUGH

class PassThrough {

    public:

    //Constructor takes in axis, min, max
    PassThrough(sl::Mat gpu_cloud, char axis, float min, float max);

    //Main processing function
    void run();

    private:

    float min;
    float max;

    //0=X 1=Y 2=Z
    int *axis;

    GPU_Cloud gpu_cloud;

}

#endif