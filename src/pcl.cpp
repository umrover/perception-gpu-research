#include <iostream>
#include <vector>
#include <algorithm>
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
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

#include "pcl.hpp"

using namespace std;
#define PCD_FOLDER "../data"

vector<string> pcd_names;

shared_ptr<pcl::visualization::PCLVisualizer> pclViewer; // = pointcloud.createRGBVisualizer();
pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>);

void readData() {
	//Long-winded directory opening (no glob, sad)
	DIR * pcd_dir;
	pcd_dir = opendir(PCD_FOLDER);
	if (NULL == pcd_dir) std::cerr<<"Input folder not exist\n";    
	
	struct dirent *dp = NULL;
	do{
    	if ((dp = readdir(pcd_dir)) != NULL) {
      		std::string file_name(dp->d_name);
      		std::cout<<"file_name is "<<file_name<<std::endl;
      		// the lengh of the tail str is at least 4
      		if (file_name.size() < 5) continue;
      		pcd_names.push_back(file_name);
      
    	}
 	} while (dp != NULL);
	std::sort(pcd_names.begin(), pcd_names.end());

}

void setPointCloud(int i) {
 	std::string pcd_name = pcd_names[i];
 	std::string full_path = PCD_FOLDER + std::string("/") + pcd_name;
	cout << "[" << i << "] " << "Loading " << full_path << endl;
  	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (full_path, *pc) == -1){ //* load the file 
    	PCL_ERROR ("Couldn't read file test_pcd.pcd \n"); 
  	}
	cout << "> Loaded point cloud of " << pc->width << "*" << pc->height << endl;
}

shared_ptr<pcl::visualization::PCLVisualizer> createRGBVisualizer(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &pt_cloud_ptr) {
    // Open 3D pclViewer and add point cloud
    shared_ptr<pcl::visualization::PCLVisualizer> pclViewer(
      new pcl::visualization::PCLVisualizer("PCL 3D pclViewer")); //This is a smart pointer so no need to worry ab deleteing it
    pclViewer->setBackgroundColor(0.12, 0.12, 0.12);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(pt_cloud_ptr);
    pclViewer->addPointCloud<pcl::PointXYZRGB>(pt_cloud_ptr, rgb);
    pclViewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5);
    pclViewer->addCoordinateSystem(1.0);
    pclViewer->initCameraParameters();
    pclViewer->setCameraPosition(0,0,-800,0,-1,0);
    return (pclViewer);
}


void ransacCPU() {
	pcl::ScopeTime t1("ransac");
	pcl::SampleConsensusModelPlane<pcl::PointXYZRGB>::Ptr model(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGB> (pc));
	pcl::RandomSampleConsensus<pcl::PointXYZRGB> ransac (model);

	pcl::PointXYZRGB min, max;

	pcl::getMinMax3D(*pc, min, max);
	cout << "min: " << min.x << ", " << min.y << ", " << min.z << endl;
	cout << "max: " << max.x << ", " << max.y << ", " << max.z << endl;

	
    ransac.setDistanceThreshold(1);
    ransac.computeModel();
	std::vector<int> inliers;
    ransac.getInliers(inliers);

	for(int i = 0; i < (int)inliers.size(); i++) {
        pc->points[inliers[i]].r = 255;
		pc->points[inliers[i]].g = 0;
        pc->points[inliers[i]].b = 0;
	}

	//pcl::copyPointCloud (*cloud, inliers, *final);

} 