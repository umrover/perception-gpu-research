#pragma once

#include "common.hpp"


#ifndef PASS_THROUGH
#define PASS_THROUGH

class PassThrough {

    public:

    //Constructor takes in axis, min, max
    PassThrough(char axis, float min, float max);

    //Main processing function
    void run(GPU_Cloud_F4 cloud);

    private:

    float min;
    float max;

    //0=X 1=Y 2=Z
    int axis;

};

#endif