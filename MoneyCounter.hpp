#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <queue>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sstream>

using namespace cv;

class Area{
public:
	Area(){}
	void add(int x, int y, int w, int h);
	int size();
	std::vector<Point2f> get(int i);
private:
	std::vector<std::vector<Point2f> > areas;
};

class MoneyCounter
{
public:
	MoneyCounter(string pattern_folder, std::string scene_img);
	~MoneyCounter();

	void set_detector(FeatureDetector* detector);
	void set_extractor(DescriptorExtractor* extractor);
	void set_matcher(DescriptorMatcher* matcher);

	void count();

	bool DEBUG_MODE;

private:
	std::queue<std::string> patterns;

	Mat img_object;
	Mat img_scene;

	FeatureDetector* detector;
	DescriptorExtractor* extractor;
	DescriptorMatcher* matcher;
	

	std::vector< DMatch > matches, good_matches;
	std::vector<KeyPoint> keypoints_object;
	std::vector<KeyPoint> keypoints_scene;
	Mat descriptors_object;
	Mat descriptors_scene;
	Mat img_matches;	

	double max_dist, min_dist;

	Mat inliers;
	Mat homography;

	std::vector<Point2f> obj_corners;
	std::vector<Point2f> scene_corners;
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	double x_shift;
	int total_value,bill_value;
	int bill_nr;
	std::string pattern_folder_sanitized;

	/* ------------------------------------------------ */
	void create_bill_list(std::string pattern_folder);
	void read_pattern();
	void filter_keypoints(std::string bill);
	void filter_keypoints_5f();
	void filter_keypoints_5b();
	void filter_keypoints_10f();
	void filter_keypoints_10b();
	void filter_keypoints_20f();
	void filter_keypoints_20b();
	void filter_keypoints_50f();
	void filter_keypoints_50b();
	void filter(Area areas);
	bool detect_bills();
	void display();
	void highlight_bill(const std::vector<Point2f> &corners, int value, double shift = 0);
};

void writeText(Mat& dst,int x,int y,string text,double scale,bool center);

