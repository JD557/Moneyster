#include "MoneyCounter.hpp"
#include <stdexcept>

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

//Detectors
FeatureDetector* d_fast = new FastFeatureDetector();
FeatureDetector* d_surf = new SurfFeatureDetector();
FeatureDetector* d_sift = new SiftFeatureDetector();
FeatureDetector* d_orb = new OrbFeatureDetector();
//Extractors
DescriptorExtractor* e_surf = new SurfDescriptorExtractor();
DescriptorExtractor* e_sift = new SiftDescriptorExtractor();
DescriptorExtractor* e_orb = new OrbDescriptorExtractor();
DescriptorExtractor* e_brief = new BriefDescriptorExtractor();
DescriptorExtractor* e_freak = new FREAK();
//Matchers
DescriptorMatcher* m_flann = new FlannBasedMatcher();
DescriptorMatcher* m_bf = new BFMatcher();

bool benchmark_mode = false;
bool debug_mode = false;

void readme();
void benchmark();
void print_benchmark(int found, int time, double precision, double recall);

int main( int argc, char** argv )
{
	int found = 0;
	string train;
	string test;


	for (int i = 1; i < argc; ++i){
		if (strcmp(argv[i], "-d")==0){
			debug_mode = true;
		}
		else if (strcmp(argv[i], "--benchmark") == 0){
			benchmark_mode = true;
		}
		else if(found<3 && !benchmark_mode){
			++found;
			if (found == 1)
				train = argv[i];
			else if (found == 2)
				test = argv[i];
			else
				break;
		}
	}

	if ((found == 0 & !benchmark_mode) || (benchmark_mode && found>0)){
		readme(); return -1;
	}

	if (benchmark_mode){
		benchmark();
	} else {

		try{
			MoneyCounter mc(train, test);
			mc.DEBUG_MODE = debug_mode;

			mc.set_detector(d_surf);
			mc.set_extractor(e_surf);
			mc.set_matcher(m_bf);
			mc.count();
			print_benchmark(mc.get_found(), mc.get_time(), mc.get_precision(), mc.get_recall());
		}
		catch (std::runtime_error e){
			std::cout << e.what() << std::endl;
			return -1;
		}

	}

	return 0;
}

struct model{
	FeatureDetector* feature;
	DescriptorExtractor* extractor;
	DescriptorMatcher* matcher;
	std::string name;
};

void readme()
	{ std::cout << " Usage: \n./MoneyCounter <bills_folder> <scene_image>\n./MoneyCounter --benchmark" << std::endl; }

void print_benchmark(int found, int time, double precision, double recall){
	std::cout << found << "," << time << "," << precision << "," << recall << std::endl;
}

void benchmark(){
	String test[] = { "./test/10s.png", "./test/5020.JPG", "./test/all.png" };
	model sets[] = { 
		{ d_surf, e_surf, m_bf, "SURF+SURF+BF"},
		//{ d_fast, e_sift, m_bf },
		{ d_sift, e_sift, m_bf, "SIFT+SIFT+BF" }/*,
		{ d_surf, e_surf, m_bf },
		{ d_surf, e_surf, m_bf },
		{ d_surf, e_surf, m_bf },
		{ d_surf, e_surf, m_bf },
		{ d_surf, e_surf, m_bf }*/
	};

	for (int i = 0; i < (sizeof(test) / sizeof(*test)); ++i){

		MoneyCounter mc("./train", test[i]);
		mc.BENCHMARK_MODE = true;
		mc.DEBUG_MODE = debug_mode;

		std::cout << "\nimage" << i+1 << ": " << test[i];

		for (int j = 0; j < (sizeof(sets) / sizeof(*sets)); ++j){
			std::cout << std::endl << sets[j].name << std::endl;
			mc.set_methods(sets[j].name);
			mc.set_detector(sets[j].feature);
			mc.set_extractor(sets[j].extractor);
			mc.set_matcher(sets[j].matcher);
			mc.count();
			print_benchmark(mc.get_found(), mc.get_time(), mc.get_precision(), mc.get_recall());
		}

	}
	waitKey();
	
}

