#pragma once

#include <Eigen/Dense>
#include "common.hpp"

using namespace Eigen;


class RansacPlane {
    public:
        /*
        REQUIRES: 
        - Zed point cloud allocated on GPU
        - Axis perpendicular to the desired plane 
        - How far off angle the found plane can be from that axis
        - Maximum number of iterations to find the plane
        EFFECTS:
        - Computes a plane perpendicular to the given axis within the tolerance that fits 
        the most data using the RANSAC algorithm
        */
        RansacPlane(Mat pointcloud, Vector3d axis, float epsilon, int iterations);

        /*
        EFFECTS:
        - Computes the RANSAC model on the GPU and returns the coefficients 
        */
        Plane computeModel();


        /*
        EFFECTS:
        - Gets GPU pointer for indicies of the inliers of the model
        */
        int* getInliers() {

        }

        /*
        - Plane equation in standard form
        */
        struct Plane {
            float a;
            float b; 
            float c;

        };

    private:
        GPU_Cloud data;
        

};