#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/time.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/PointIndices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

#define PCD_FOLDER "/data/"


shared_ptr<pcl::visualization::PCLVisualizer> viewer; // = pointcloud.createRGBVisualizer();
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc;

void readData() {
	
	//Read in image names
 	std::string pcd_name = pcd_names[idx_curr_pcd_img];
 	std::string full_path = PCD_FOLDER + std::string("/") + pcd_name;
  	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (full_path, *p_pcl_point_cloud) == -1){ //* load the file 
    	PCL_ERROR ("Couldn't read file test_pcd.pcd \n"); 
  	} 
}

int main() {
	

	cout << "hi" << endl;
	return 0;
}
