/*
The code is referenced from git@github.com:parkseobin/resnet34_mkl.git
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <ctime>
#include "cnpy.h"
#include "mkl.h"
#include "resnet.hpp"
// use pure mkl
// compile by Makefile
using namespace std;

void test_resnet(float x_, float y_, float w_, float h_);

	
int main(int argc, char* argv[]){

	if(argc < 5){
		cout << "need parameters" << endl;
		return 0;
	}

	// ------------------------------------------
	// some tests
	init_settings();
	load_parameter("test.npz", 0);

	test_resnet(::atof(argv[1]), ::atof(argv[2]), ::atof(argv[3]), ::atof(argv[4]));
	free_parameter(0);
	// ------------------------------------------

	return 0;
}


void test_resnet(float x_, float y_, float w_, float h_){
		

	cv::Mat image;
	image = cv::imread("input.jpg",  cv::IMREAD_COLOR);

	float* in_mat = (float*)mkl_malloc(224*224*3*sizeof(float), 32);
	preprocess(image, in_mat, x_, y_, w_, h_);
	//preprocess(image, in_mat, 0, 0, 1, 1);

	cout << "start resnet" << endl;
	clock_t begin = clock();

	float out_prob = pass_RESNET(in_mat, 0);

	clock_t end = clock();
	float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
	cout << "resnet end\ntime: " << elapsed_secs << endl;
	cout << "probability: " << out_prob << endl;

	return;
}