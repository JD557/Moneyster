#include "MoneyCounter.hpp"

/*
 * Calculate keypoints and descriptors (related vector <-> mat )
 * Store and create a temporary working of above ( to restore later on )
 * Calculate good matches
 * Calculate Homography and inliers(Mat)
 * foreach row of the inliers Mat, if the first element==1 then the good_matches[#row] is an inlier 
 * check if all inliers on good_matches are insire the area of the note ( pointPolygonTest(scene_corners, X), X=> single inlier value from the filtered good_matches)
 *  if all are inside, its a good note => remove all the matches inside that area ( either inliers or not ) => loop
 *  if at least one of them is inside OR the number of matches is inferior to 4 ( the minimum for a Homography ), there are no more notes
 * 
 * to detect other kinds of bill, restor the keypoints and descriptors stored at first and apply the same algorithm with a different bill image
 */


void readme();

int main( int argc, char** argv )
{
	if (argc != 3) { readme(); return -1; }

	try{
		MoneyCounter mc = MoneyCounter(argv[1], argv[2]);
		mc.DEBUG_MODE = false;
		mc.set_detector(new SurfFeatureDetector(400));
		mc.set_extractor(new SurfDescriptorExtractor());
		mc.set_matcher(new BFMatcher());
		mc.count();
	}
	catch (std::runtime_error e){
		std::cout << e.what() << std::endl;
		return -1;
	}

	return 0;
}


void readme()
	{ std::cout << " Usage: ./MoneyCounter <bills_folder> <scene_image>" << std::endl; }