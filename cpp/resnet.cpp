/*
The code is referenced from git@github.com:parkseobin/resnet34_mkl.git
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <ctime>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "mkl.h"
#include "cnpy.h"
#include "resnet.hpp"
// use pure mkl
// compiling: my Makefile
using namespace std;

cnpy::npz_t* NPZ_FILE;
int PARAM_NUMBER = 5;
float*** RESNET_PARAM = NULL;



void init_settings(){
	RESNET_PARAM = (float***)malloc(PARAM_NUMBER*sizeof(float**));
	NPZ_FILE = (cnpy::npz_t*)malloc(PARAM_NUMBER*sizeof(cnpy::npz_t));
	int i;
	for(i=0; i<PARAM_NUMBER; i++){
		RESNET_PARAM[i] = NULL;
	}
}

float pass_RESNET(float* in_mat, int param_ind){
	/*******************************
		@in_mat: normalized matrix (0~1)


	*******************************/
	
	float *out_mat, *W, *B;
	// layer0
	W = fetch_parameters(0, 0, 0, 0, param_ind);
	out_mat = (float*)mkl_malloc(112*112*64*sizeof(float), 32);
	pass7x7convolution_with_stride(in_mat, out_mat, W, 224, 224, 3, 64);

	W = fetch_parameters(0, 0, 0, 1, param_ind);
	B = fetch_parameters(0, 0, 0, 2, param_ind);
	batch_normalization(out_mat, W, B, 112, 112, 64);

	relu(out_mat, 112*112*64);
	pooling(out_mat, 112, 112, 64, 0);

	// layer1
	out_mat = pass_basic_block(out_mat, 1, 0, 56, 56, 64, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 1, 1, 56, 56, 64, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 1, 2, 56, 56, 64, 0, param_ind);

	// layer2
	out_mat = pass_basic_block(out_mat, 2, 0, 28, 28, 128, 1, param_ind);
	out_mat = pass_basic_block(out_mat, 2, 1, 28, 28, 128, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 2, 2, 28, 28, 128, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 2, 3, 28, 28, 128, 0, param_ind);

	// layer3
	out_mat = pass_basic_block(out_mat, 3, 0, 14, 14, 256, 1, param_ind);
	out_mat = pass_basic_block(out_mat, 3, 1, 14, 14, 256, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 3, 2, 14, 14, 256, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 3, 3, 14, 14, 256, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 3, 4, 14, 14, 256, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 3, 5, 14, 14, 256, 0, param_ind);

	// layer4
	out_mat = pass_basic_block(out_mat, 4, 0, 7, 7, 512, 1, param_ind);
	out_mat = pass_basic_block(out_mat, 4, 1, 7, 7, 512, 0, param_ind);
	out_mat = pass_basic_block(out_mat, 4, 2, 7, 7, 512, 0, param_ind);

	pooling(out_mat, 7, 7, 512, 1);
	
	// fc layer!
	W = fetch_parameters(5, 0, 0, 0, param_ind);
	B = fetch_parameters(5, 0, 0, 1, param_ind);
	float* y_out = (float*)mkl_malloc(2*sizeof(float), 32);
	y_out[0] = B[0];
	y_out[1] = B[1];
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
				2, 1, 512, 1.0f, W, 512, out_mat, 1, 1.0f, y_out, 1);
	
	float last_output = 1.0f / (1 + exp(y_out[0] - y_out[1]));
	mkl_free(out_mat);
	mkl_free(y_out);

	return last_output;
}

float* pass_basic_block(float* in_mat, int layer_num, int block_num, 
					  int w, int h, int c, int strided, int param_ind){
	/*******************************

		- conv(3x3)
		- BN
		- relu
		- conv(3x3)
		- BN
		- residual add (with conv)
		- relu

		@ in_mat: (hxw x c) matrix
		@ strided: 
		true if first conv has stride=2.
		strided convolution is applied to the residual too.

	*******************************/

	float *W, *B;

	// conv1
	W = fetch_parameters(layer_num, block_num, 0, 0, param_ind);
	float *conv1;
	if(strided){
		conv1 = (float*)mkl_malloc(h*w*c*sizeof(float), 32);
		pass3x3convolution(in_mat, conv1, W, w*2, h*2, c/2, c, 2);
	}else{
		conv1 = (float*)mkl_malloc(h*w*c*sizeof(float), 32);
		pass3x3convolution(in_mat, conv1, W, w, h, c, c, 1);
	}

	// residual convolution, bn
	float* res;
	if(strided){
		res = (float*)mkl_malloc(w*h*c*sizeof(float), 32);
		W = fetch_parameters(layer_num, block_num, 0, 3, param_ind);
		pass1x1convolution_downsample(in_mat, res, W, w, h, c);
		W = fetch_parameters(layer_num, block_num, 0, 4, param_ind);
		B = fetch_parameters(layer_num, block_num, 0, 5, param_ind);
		batch_normalization(res, W, B, w, h, c);
	}else{
		res = (float*)mkl_malloc(w*h*c*sizeof(float), 32);
		cblas_scopy(w*h*c, in_mat, 1, res, 1);
	}
	mkl_free(in_mat);

	// bn1
	W = fetch_parameters(layer_num, block_num, 0, 1, param_ind);
	B = fetch_parameters(layer_num, block_num, 0, 2, param_ind);
	batch_normalization(conv1, W, B, w, h, c);

	// relu
	relu(conv1, w*h*c);

	// conv1
	W = fetch_parameters(layer_num, block_num, 1, 0, param_ind);
	float* conv2 = (float*)mkl_malloc(w*h*c*sizeof(float), 32);
	pass3x3convolution(conv1, conv2, W, w, h, c, c, 1);

	// bn2
	W = fetch_parameters(layer_num, block_num, 1, 1, param_ind);
	B = fetch_parameters(layer_num, block_num, 1, 2, param_ind);
	batch_normalization(conv2, W, B, w, h, c);

	// residual addition 
	// with GEMM?
	int i; 
	for(i=0; i<w*h*c; i++){
		conv2[i] += res[i];
	}

	
	// relu
	relu(conv2, w*h*c);
	mkl_free(res);
	mkl_free(conv1);
	return conv2;
}

void load_parameter(const char* filename, int param_ind){
	/*******************************

		loading parameters from filename

	*******************************/

	cnpy::npz_t tmp_npz = cnpy::npz_load(filename);
	NPZ_FILE[param_ind] = tmp_npz;
	RESNET_PARAM[param_ind] = (float**)malloc(120*sizeof(float*));
	cnpy::NpyArray arr;

	int i, j, k;
	int layer_len[] = {3, 4, 6, 3};
	int depth[] = {64, 128, 256, 512};
	int count = 0;
	string name;


	// From the first pooling to the last pooling (count=105)

	// layer number
	for(i=0; i<4; i++){
		// block number
		for(j=0; j<layer_len[i]; j++){
			name = "layer" + to_string(i+1) + "." + to_string(j) + ".";
			arr = tmp_npz[name + "conv1.weight"];
			RESNET_PARAM[param_ind][count++] = arr.data<float>();
			if(i!=0 && j==0){	
				transpose_weight(RESNET_PARAM[param_ind][count-1], 3, 3, depth[i]/2, depth[i]);
			}else{
				transpose_weight(RESNET_PARAM[param_ind][count-1], 3, 3, depth[i], depth[i]);
			}
			arr = tmp_npz[name + "bn1.weight"];
			RESNET_PARAM[param_ind][count++] = arr.data<float>();
			arr = tmp_npz[name + "bn1.bias"];
			RESNET_PARAM[param_ind][count++] = arr.data<float>();
			arr = tmp_npz[name + "conv2.weight"];
			RESNET_PARAM[param_ind][count++] = arr.data<float>();
			transpose_weight(RESNET_PARAM[param_ind][count-1], 3, 3, depth[i], depth[i]);
			arr = tmp_npz[name + "bn2.weight"];
			RESNET_PARAM[param_ind][count++] = arr.data<float>();
			arr = tmp_npz[name + "bn2.bias"];
			RESNET_PARAM[param_ind][count++] = arr.data<float>();
			// include downsample
			if(i!=0 && j==0){
				arr = tmp_npz[name + "downsample.0.weight"];
				RESNET_PARAM[param_ind][count++] = arr.data<float>();
				transpose_downsample(RESNET_PARAM[param_ind][count-1], depth[i]);
				arr = tmp_npz[name + "downsample.1.weight"];
				RESNET_PARAM[param_ind][count++] = arr.data<float>();
				arr = tmp_npz[name + "downsample.1.bias"];
				RESNET_PARAM[param_ind][count++] = arr.data<float>();
			}
		}
	}

	// layer0
	arr = tmp_npz["conv1.weight"];
	RESNET_PARAM[param_ind][count++] = arr.data<float>();
	transpose_weight(RESNET_PARAM[param_ind][count-1], 7, 7, 3, 64);
	arr = tmp_npz["bn1.weight"];
	RESNET_PARAM[param_ind][count++] = arr.data<float>();
	arr = tmp_npz["bn1.bias"];
	RESNET_PARAM[param_ind][count++] = arr.data<float>();

	// fc layer
	arr = tmp_npz["fc.weight"];
	RESNET_PARAM[param_ind][count++] = arr.data<float>();
	arr = tmp_npz["fc.bias"];
	RESNET_PARAM[param_ind][count++] = arr.data<float>();

	return;
}

float* fetch_parameters(int layer_num, int block_num, 
						int conv_num, int type, int param_ind){
	/*******************************

		Fetching parameters

		@ layer_num: first index
		(layer0 is 7x7 conv)
		@ block_num: second index
		@ conv_num: third index
		@ type: 
			0 >> conv weight
			1 >> bn weight
			2 >> bn bias
			3 >> downsample conv weight
			4 >> downsample bn weight
			5 >> downsample bn bias

	*******************************/
	if(RESNET_PARAM[param_ind] == NULL){
		// panic!!
		return NULL;
	}

	float* output;

	if(layer_num == 0){
		// 7x7 conv
		output = RESNET_PARAM[param_ind][105 + type];

	}else if(layer_num <= 4){
		int start_point[] = {0, 18, 45, 84};		
		if(type > 2){
			conv_num = 1;
		}
		int index = start_point[layer_num-1] 
					+ 6*block_num + 3*conv_num + type;
		if(block_num > 0 && layer_num > 1){
			index += 3;
		}
		output = RESNET_PARAM[param_ind][index];

	}else if(layer_num == 5){
		// fully connected layer
		// type = 0 for weight
		// 		  1 for bias
		output = RESNET_PARAM[param_ind][108 + type];
	}else{
		// panic!
	}

	return output;
}


void pooling(float* in_mat, int w, int h, int c, int type){
	/*******************************

		max/avg pooling
		height and width will be halved
		(need to be checked)

		@ type: 
			0 >> max
			1 >> avg

	*******************************/

	int i, j, k, l, m, index;
	float max, sum;

	if(type == 0){
		// max pooling
		// kernel_size=3, stride=2, padding=1
		float* tmp = (float*)mkl_malloc(h*w*c*sizeof(float), 32);
		cblas_scopy(h*w*c, in_mat, 1, tmp, 1);
		for(i=0; i<c; i++){
			for(j=0; j<h; j+=2){
				for(k=0; k<w; k+=2){
					max = 0.0f;
					for(l=-1; l<2; l++){
					for(m=-1; m<2; m++){
						index = i + c*(h*(k+l) + (j+m));
						if(k+l >=0 && k+l < w &&
						   j+m >=0 && j+m < h && tmp[index] > max)
							max = tmp[index];
					}
					}
					in_mat[i + c*(h/2*k/2 + j/2)] = max;
				}
			}
		}
		mkl_free(tmp);
	}else if(type == 1){
		// avg pooling just for resnet's last layer

		for(k=0; k<c; k++){
			sum = 0.0f;
			for(i=0; i<w; i++){
				for(j=0; j<h; j++){
					sum += in_mat[k + c*(j + i*h)];
				}
			}
			in_mat[k] = sum / 49;
		}
	}else{
		// panic
	}

	return;
}


void batch_normalization(float* in_mat, float* W, float* B,
						 int w, int h, int c){
	/*******************************

	  BN using raw GEMM
	  this should be modified.
	  .000319 on layer1
	  
	  @ W: scaling matrix of dimension cxc
	  @ B: bias same dimension as in_mat

	*******************************/

	float* tmp = (float*)mkl_malloc(w*h*c*sizeof(float), 32);
	MKL_INT incx = 1;
	cblas_scopy(w*h*c, B, incx, tmp, incx);

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			h*w, c, c, 1.0f, in_mat, c, W, c, 1.0f, tmp, c);

	cblas_scopy(w*h*c, tmp, incx, in_mat, incx);
	mkl_free(tmp);

	return;
}

void pass7x7convolution_with_stride(float* in_mat, float* out_mat,
							float *W, int w, int h, int c, int c_out){
	// https://software.intel.com/en-us/mkl-developer-reference-c-fft-code-examples
	/*******************************

		passing 7x7 convolution

	*******************************/

	int stride = 2;
	int m, n, k;
	m = w*h/(stride*stride);
	k = 49*c;
	n = c_out;

	float *A, *B, *C;
	A = in_mat;
	B = (float*)mkl_malloc(m*k*sizeof(float), 32);
	
	make_patch(A, B, w, h, c, 7, 7, stride);

	C = out_mat;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, 1.0f, B, k, W, n, 0.0f, C, n);
	mkl_free(B);
	return;
}

void pass1x1convolution_downsample(float* in_mat, float* out_mat,
								   float* W, int w, int h, int c){
	/*******************************

		passing 1x1 convolution with downsampling
		for residuals

		the output h, w will be halved, 
		and output c will be doubled.

	*******************************/

	float* B = (float*)mkl_malloc(h*w*c/2*sizeof(float), 32);
	int i, j, k;
	for(i=0; i<w; i++){
		for(j=0; j<h; j++){
			for(k=0; k<c/2; k++){
				B[k + c/2*(j + h*i)] = in_mat[k + c/2*(j*2 + 2*h*i*2)];
			}
		}
	}

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			h*w, c, c/2, 1.0f, B, c/2, W, c, 0.0f, out_mat, c);
	mkl_free(B);
	return;
}

void pass3x3convolution(float* in_mat, float* out_mat,
						float* W, int w, int h, 
						int c, int out_c, int stride){
	/*******************************

	  @ in_mat: (hxw x c) matrix
	  @ out_mat: (hxw/stride^2 x c) matrix
	  @ W: weights
	  @ h, w, c: input dimensions
	  @ out_c: output channel

	*******************************/

	int m, n, k;
	m = w*h/(stride*stride);
	k = 9*c;
	n = out_c;

	float *A, *B, *C;
	A = in_mat;
	B = (float*)mkl_malloc(m*k*sizeof(float), 32);
	
	make_patch(A, B, w, h, c, 3, 3, stride);

	C = out_mat;
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			m, n, k, 1.0f, B, k, W, n, 0.0f, C, n);
	mkl_free(B);
	return;
}


void patch_operation(float* in_mat, float* out_mat, 
					 int b1, int b2, int b3, int i, int j, 
					 int w, int h, int c, 
					 int wp, int hp, int stride){

	int beta = hp*wp*c*(h/stride*i/stride + j/stride) + b3 + c*(hp*b1 + b2);
	if(i-wp/2+b1 < 0 || i-hp/2+b1 >= w ||
	   j-wp/2+b2 < 0 || j-hp/2+b2 >= h){
		out_mat[beta] = 0.0f;
		return;
	}

	int alpha = b3 + c*(h*(i-wp/2+b1) + (j-hp/2+b2));
	out_mat[beta] = in_mat[alpha];
	return;
}

void make_patch(float* in_mat, float* out_mat, 
				  int w, int h, int c,
				  int wp, int hp, int stride){
	/*******************************

		This function should be optimized!!

	*******************************/

	int i, j, b1, b2, b3;
	for(i=0; i<w; i+=stride){
		for(j=0; j<h; j+=stride){
			for(b1=0; b1<wp; b1++){
				for(b2=0; b2<hp; b2++){
					for(b3=0; b3<c; b3++){
						patch_operation(in_mat, out_mat, 
								b1, b2, b3, i, j, 
								w, h, c, wp, hp, stride);
					}
				}
			}
		}
	}

	return;
}


void relu(float* in, int size){
	int i;
	for(i=0; i<size; i++){
		if(in[i] < 0){	
			in[i] = 0;
		}
	}

	return;
}

void transpose_weight(float* in_mat, int w, int h, int c, int c_out){
	
	float* tmp = (float*)mkl_malloc(w*h*c*c_out*sizeof(float), 32);

	int i, j, k, l;
	cblas_scopy(w*h*c*c_out, in_mat, 1, tmp, 1);
	for(i=0; i<w; i++){
		for(j=0; j<h; j++){
			for(k=0; k<c; k++){
				for(l=0; l<c_out; l++){
					// (j, i, k, l) -> (i, j, k, l)
					in_mat[l + c_out*(k + c*(j + h*i))] 
						 = tmp[l + c_out*(k + c*(i + h*j))];
				}
			}	
		}
	}
	mkl_free(tmp);
	return;
}

void transpose_downsample(float* in_mat, int c){

	// transpose weight
	float* tmp = (float*)mkl_malloc(c*c/2*sizeof(float), 32);
	int i, j;
	cblas_scopy(c*c/2, in_mat, 1, tmp, 1);
	for(i=0; i<c; i++){
		for(j=0; j<c/2; j++){
			in_mat[c*j + i] = tmp[j + i*c/2];
		}
	}
	mkl_free(tmp);
	return;
}

void transpose_img(float* in_mat){

	int w=224, h=224, c=3;
	float* tmp = (float*)mkl_malloc(w*h*c*sizeof(float), 32);
	int i, j, k;
	cblas_scopy(w*h*c, in_mat, 1, tmp, 1);
	for(i=0; i<w; i++){
		for(j=0; j<h; j++){
			for(k=0; k<c; k++){
				// (h, w, c) -> (w, h, c)
				in_mat[k + c*(i + w*j)]
					= tmp[k + c*(j + h*i)];
			}
		}
	}

	mkl_free(tmp);
	return;
}

void enhance_dark_image(uint8_t* in_mat){
	/*******************************

		@in_mat: - this matrix is not normalized (0~255 integer)
				 - it has depth of 3 (grayscale, but rgb)

	*******************************/
	int cdf[256];
	int i, tmp;
	int h=224, w=224;

	// initialize cdf
	for(i=0; i<256; i++){
		cdf[i] = 0;
	}

	// make cdf of pixels
	for(i=0; i<w*h; i++){
		// *3 for the depth
		cdf[in_mat[i*3]]++;
	}
	for(i=1; i<256; i++){
		cdf[i] += cdf[i-1];
	}

	// calculate pixels
	for(i=0; i<w*h; i++){
		tmp = cdf[in_mat[i*3]] - cdf[0];
		in_mat[i*3] = tmp * 255 / (h*w - cdf[0]);
		in_mat[i*3 + 1] = in_mat[i*3];
		in_mat[i*3 + 2] = in_mat[i*3];
	}

	return;
}

void preprocess(cv::Mat image, float* in_mat, float x_, float y_, float w_, float h_){
	/*******************************

		squash and normalize

		@ x_, y_, w_, h_ are between 0~1
		@ x_+w_ and y_+h_ cannot exceed 1


		*some of the codes are removed*

	*******************************/
	cv::Mat square;
	cv::resize(image, square, cv::Size(224, 224));

	int i;
	for(i=0; i<224*224*3; i++){
		in_mat[i] = ((float)square.data[i]) / 255.0f - 0.5126f;
	}
	transpose_img(in_mat);

	return;
}


void free_parameter(int param_ind){
	if(RESNET_PARAM[param_ind]){
		free(RESNET_PARAM[param_ind]); 
		RESNET_PARAM[param_ind] = NULL;
		NPZ_FILE[param_ind].clear();
	}

	return;
}
