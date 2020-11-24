#include <Eigen/Dense>
 
using namespace Eigen;


class RansacPlane {
    public:
        RansacPlane(Mat pointcloud, Vector3d axis, float epsilon);
        void computeModel();

    private:


};