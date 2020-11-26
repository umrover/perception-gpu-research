#pragma once

#include <Eigen/Dense>
#include "common.hpp"

using namespace Eigen;


class RansacPlane {
    public:
        struct Plane {
            float a;
            float b; 
            float c;
            float d;

        };

        /*
        REQUIRES: 
        - Zed point cloud allocated on GPU
        - Axis perpendicular to the desired plane 
        - How far off angle the found plane can be from that axis
        - Maximum number of iterations to find the plane
        - Maximum distance from the plane to be considered an inlier
        EFFECTS:
        - Computes a plane perpendicular to the given axis within the tolerance that fits 
        the most data using the RANSAC algorithm
        */
        RansacPlane(GPU_Cloud pc, Vector3d axis, float epsilon, int iterations, float threshold);

        /*
        EFFECTS:
        - Computes the RANSAC model on the GPU and returns the coefficients 
        */
        Plane computeModel();


        /*
        EFFECTS:
        - Gets GPU pointer for indicies of the inliers of the model
        */
        int* getInliers();

        /*
        - Plane equation in standard form
        */


    private:
        GPU_Cloud pc;
        GPU_Indicies inliers;
        Vector3d axis;
        float epsilon;
        int iterations;
        float threshold;

};