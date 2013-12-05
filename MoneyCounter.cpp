#include "MoneyCounter.hpp"
#define FONT_FACE cv::FONT_HERSHEY_SCRIPT_COMPLEX
#define FONT_THICKNESS 3
#define FONT_RATIO 4
using std::string;
using std::cout; 
using std::endl; 

MoneyCounter::MoneyCounter(string pattern_folder, string scene_img){
	create_bill_list(pattern_folder);

	img_scene = imread(scene_img, CV_LOAD_IMAGE_GRAYSCALE);
	if (!img_scene.data) throw std::runtime_error("Missing image");

	total_value = 0;
	obj_corners.resize(4);
	img_matches = imread(scene_img);

	DEBUG_MODE = false;
}

MoneyCounter::~MoneyCounter()
{
/*	delete detector;
	delete extractor;
	delete matcher; */
}

void MoneyCounter::create_bill_list(string pattern_folder){
	
	char l = pattern_folder[pattern_folder.size() - 1];
	if (l == '\\' || l == '/'){
		pattern_folder_sanitized = pattern_folder.substr(0, pattern_folder.size() - 2) + '/';
	}
	else {
		pattern_folder_sanitized = pattern_folder + '/';
	}

	patterns.push("5f.jpg");
	patterns.push("5b.jpg");
	patterns.push("10f.jpg");
	patterns.push("10b.jpg");
	patterns.push("20f.jpg");
	patterns.push("20b.jpg");
	patterns.push("50f.jpg");
	patterns.push("50b.jpg");
}

void MoneyCounter::set_detector(FeatureDetector* detector){
	this->detector = detector;
}

void MoneyCounter::set_extractor(DescriptorExtractor* extractor){
	this->extractor = extractor;
}

void MoneyCounter::set_matcher(DescriptorMatcher* matcher){
	this->matcher = matcher;
}

void MoneyCounter::count() {
	if (detector == NULL) 
		throw std::runtime_error("Missing detector implementation");
	if (extractor == NULL)
		throw std::runtime_error("Missing extractor implementation");
	if (matcher == NULL)
		throw std::runtime_error("Missing matcher implementation");


	detector->detect(img_scene, keypoints_scene);

	if (keypoints_scene.size() == 0){
		display();
		return;
	}

	extractor->compute(img_scene, keypoints_scene, descriptors_scene);



	std::vector<KeyPoint> keypoints_scene_bk = keypoints_scene;
	Mat descriptors_scene_bk = descriptors_scene.clone();

	while (!patterns.empty()){
		read_pattern();
		while (detect_bills());
		keypoints_scene = keypoints_scene_bk;
		descriptors_scene = descriptors_scene_bk.clone();
	}

	display();
}

void MoneyCounter::display(){

	//cv::Rect bounding_rect = cv::boundingRect();
	//cv::Point2f center(bounding_rect.x + bounding_rect.width / 2.0, bounding_rect.y + bounding_rect.height / 2.0);
	//double max_dimen = bounding_rect.width > bounding_rect.height ? bounding_rect.width : bounding_rect.height;

	std::stringstream ss;
	ss << total_value;
	string text = ss.str();

	double font_scale = 1;
	int baseline = 0;
	cv::Size text_size = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);

	font_scale = img_scene.rows / (text_size.width * FONT_RATIO);
	text_size = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);
	cv::Point2f text_position =cv::Point2f(img_scene.rows*0.7, 50);
	cv::putText(img_matches, text, text_position, FONT_FACE, font_scale, cv::Scalar(255, 0, 0), FONT_THICKNESS);

	imshow("Good Matches & Object detection", img_matches);
	waitKey(0);
}

bool MoneyCounter::detect_bills(){
	good_matches = vector< DMatch >();
	matches = vector< DMatch >();
	
	matcher->match(descriptors_object, descriptors_scene, matches);
	
	max_dist = 0;
	min_dist = 100;


	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_object.rows; i++){
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	for (int i = 0; i < descriptors_object.rows; i++){
		if (matches[i].distance < 3 * min_dist){
			good_matches.push_back(matches[i]);
		}
	}

	//not enough points for a homography
	if (good_matches.size() < 4) {
		return false;
	}

	if (DEBUG_MODE)
	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>());
	
	
	obj.clear(); scene.clear();
	for (int i = 0; i < good_matches.size(); i++){
		//-- Get the keypoints from the good matches
		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	}

	homography = findHomography(obj, scene, CV_RANSAC, 3, inliers);
	perspectiveTransform(obj_corners, scene_corners, homography);


	std::vector< DMatch > good_matches_inliers;
	for (int i = 0; i < inliers.rows; ++i){
		if (inliers.at<uchar>(i,0) != 0){
			good_matches_inliers.push_back(good_matches[i]);
		}
	}

	//check if all inliers are inside the area
	for (int i = 0; i < good_matches_inliers.size(); ++i){
		if (pointPolygonTest(scene_corners, keypoints_scene[good_matches_inliers[i].trainIdx].pt, false) < 0){
			return false;
		}
	}

	if (DEBUG_MODE){
		highlight_bill(scene_corners, bill_value, img_object.cols);
		imshow("Good Matches & Object detection", img_matches);
		cv::waitKey(0);
	}
	else {
		highlight_bill(scene_corners, bill_value);
	}
	
	total_value += bill_value;
	cout << "+" << bill_value << " total:" << total_value<<endl;
	
	//remove all the matches from the area
	cv::Mat new_descriptors;
	unsigned int j = 0;
	for (unsigned int i = 0; i < keypoints_scene.size(); ++i, ++j) {
		if (pointPolygonTest(scene_corners, keypoints_scene[i].pt, false) >= 0) {
			keypoints_scene.erase(keypoints_scene.begin() + i);
			i--;
		}
		else {
			new_descriptors.push_back(descriptors_scene.row(j));
		}
	}
	descriptors_scene = new_descriptors;

	return true;
}

void MoneyCounter::read_pattern(){
	if (patterns.empty())
		throw std::runtime_error("Missing pattern");

	img_object = imread(pattern_folder_sanitized+patterns.front(), CV_LOAD_IMAGE_GRAYSCALE);
	

	if (!img_object.data) throw std::runtime_error("Missing image");

	detector->detect(img_object, keypoints_object);
		filter_keypoints(patterns.front());

	extractor->compute(img_object, keypoints_object, descriptors_object);
	

	patterns.pop();

	//x_shift = img_object.cols;
	//-- Get the corners from the image_1 ( the object to be "detected" )
	
	obj_corners[0] = Point(0, 0);
	obj_corners[1] = Point(img_object.cols, 0);
	obj_corners[2] = Point(img_object.cols, img_object.rows);
	obj_corners[3] = Point(0, img_object.rows);
}

void MoneyCounter::filter_keypoints(std::string bill){
	if (bill == "5f.jpg"){
		bill_value = 5;
		filter_keypoints_5f();
	}
	else if (bill == "5b.jpg"){
		bill_value = 5;
		filter_keypoints_5b();
	}
	else if (bill == "10f.jpg"){
		bill_value = 10;
		filter_keypoints_10f();
	}
	else if (bill == "10b.jpg"){
		bill_value = 10;
		filter_keypoints_10f();
	}
	else if (bill == "20f.jpg"){
		bill_value = 20;
		filter_keypoints_20f();
	}
	else if (bill == "20b.jpg"){
		bill_value = 20;
		filter_keypoints_20b();
	}
	else if (bill == "50f.jpg"){
		bill_value = 50;
		filter_keypoints_50f();
	}
	else if (bill == "50b.jpg"){
		bill_value = 50;
		filter_keypoints_50b();
	}
}

void MoneyCounter::filter(Area areas){
	vector<KeyPoint> temp;

	for (int j = 0; j<areas.size(); ++j)
		for (int i = 0; i < keypoints_object.size(); ++i){
			if (pointPolygonTest(areas.get(j), keypoints_object[i].pt, false) >= 0){
				temp.push_back(keypoints_object[i]);
			}
		}

	keypoints_object = temp;
}

void MoneyCounter::filter_keypoints_5f(){
	Area a;
	a.add(6, 6, 20, 24);
	a.add(8, 92, 23, 28);
	a.add(128, 4, 104, 104);
	filter(a);
}

void MoneyCounter::filter_keypoints_5b(){
	Area a;
	a.add(9,5,15,21);
	a.add(7,100,20,30);
	a.add(240,107,16,22);
	filter(a);
}

void MoneyCounter::filter_keypoints_10f(){
	Area a;
	a.add(4,4,27,24);
	a.add(5,102,33,31);
	a.add(168,6,60,52);
	filter(a);
}

void MoneyCounter::filter_keypoints_10b(){
	Area a;
	a.add(10,6,20,20);
	a.add(9, 10, 30, 27);
	a.add(233, 111, 23, 23);
	a.add(233, 4, 23, 23);
	filter(a);
}

void MoneyCounter::filter_keypoints_20f(){
	Area a;
	a.add(4, 4, 4 + 26, 4 + 23);
	a.add(6, 109, 6 + 31, 109 + 27);
	a.add(159, 5, 159 + 63, 5 + 52);
	a.add(129, 45, 129 + 106, 45 + 92);
	filter(a);
}

void MoneyCounter::filter_keypoints_20b(){
	Area a;
	a.add(6, 3, 6 + 150, 3 + 61);
	a.add(7, 111, 7 + 32, 111 + 29);
	a.add(236, 3, 236 + 24, 3 + 26);
	a.add(232, 114, 232 + 28, 114 + 25);
	filter(a);
}

void MoneyCounter::filter_keypoints_50f(){
	Area a;
	a.add(3, 5, 3 + 24, 5 + 23);
	a.add(3, 114, 3 + 32, 114 + 26);
	a.add(166, 4, 166 + 64, 4 + 52);
	a.add(142, 26, 142 + 80, 26 + 105);
	filter(a);
}

void MoneyCounter::filter_keypoints_50b(){
	Area a;
	a.add(7, 5, 7 + 160, 5 + 57);
	a.add(8, 114, 8 + 36, 114 + 29);
	a.add(237, 4, 237 + 24, 4 + 24);
	a.add(220, 111, 220 + 36, 111 + 28);
	filter(a);
}

void MoneyCounter::highlight_bill(const std::vector<Point2f> &corners, int value, double shift){
	line(img_matches, corners[0] + Point2f(shift, 0), corners[1] + Point2f(shift, 0), Scalar(0, 255, 0), 4);
	line(img_matches, corners[1] + Point2f(shift, 0), corners[2] + Point2f(shift, 0), Scalar(0, 255, 0), 4);
	line(img_matches, corners[2] + Point2f(shift, 0), corners[3] + Point2f(shift, 0), Scalar(0, 255, 0), 4);
	line(img_matches, corners[3] + Point2f(shift, 0), corners[0] + Point2f(shift, 0), Scalar(0, 255, 0), 4);

	cv::Rect bounding_rect = cv::boundingRect(corners);
	cv::Point2f center(bounding_rect.x + bounding_rect.width / 2.0, bounding_rect.y + bounding_rect.height / 2.0);
	double max_dimen = bounding_rect.width > bounding_rect.height ? bounding_rect.width : bounding_rect.height;

	std::stringstream ss;
	ss << value;
	string text = ss.str();

	double font_scale = 1;
	int baseline = 0;
	cv::Size text_size = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);

	font_scale = max_dimen / (text_size.width * FONT_RATIO);
	text_size = cv::getTextSize(text, FONT_FACE, font_scale, FONT_THICKNESS, &baseline);
	cv::Point2f text_position = center - cv::Point2f(text_size.width / 2.0, -text_size.height / 2.0);
	cv::putText(img_matches, text, text_position, FONT_FACE, font_scale, cv::Scalar(255, 0, 0), FONT_THICKNESS);

}

void Area::add(int x, int y, int w, int h){
	vector<Point2f> a;
	a.push_back(Point(x, y));
	a.push_back(Point(x+w, y));
	a.push_back(Point(x+w, y+h));
	a.push_back(Point(x, y+h));
	areas.push_back(a);
}

int Area::size(){
	return areas.size();
}

vector<Point2f> Area::get(int i){
	return areas[i];
}