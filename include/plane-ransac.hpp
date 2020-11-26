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
        RansacPlane(Vector3d axis, float epsilon, int iterations, float threshold);

        ~RansacPlane();

        /*
        EFFECTS:
        - Computes the RANSAC model on the GPU and returns the coefficients 
        */
        Plane computeModel(GPU_Cloud pc);


        /*
        EFFECTS:
        - Gets GPU pointer for indicies of the inliers of the model
        */
        GPU_Indicies getInliers();

        /*
        - Plane equation in standard form
        */



    private:
        //user given model parms
        GPU_Cloud pc;
        GPU_Indicies inliers;
        Vector3d axis;
        float epsilon;
        int iterations;
        float threshold;

        //internal info
        int* inlierCounts; 
        int* modelPoints; 
        int selection;

};