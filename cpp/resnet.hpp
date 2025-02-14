/*
The code is referenced from git@github.com:parkseobin/resnet34_mkl.git
*/

#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>


// functions in use outside
void init_settings();
void load_parameter(const char* filename, int param_ind);
void free_parameter(int param_ind);
void preprocess(cv::Mat image, float* in_mat, float x_, float y_, float w_, float h_);
float pass_RESNET(float* in_mat, int param_ind);



// ---------------
// inner functions
// ---------------
void relu(float* in, int size);
void pass1x1convolution_downsample(float* in_mat, float* out_mat,
								   float* W, int w, int h, int c);
void pass3x3convolution(float* in_mat, float* out_mat, float* W, 
					int w, int h, int c, int c_out, int stride);
void pass7x7convolution_with_stride(float* in_mat, float* out_mat,
							float *W, int w, int h, int c, int c_out);
void patch_operation(float* in_mat, float* out_mat, 
					 int b1, int b2, int b3, int i, int j, 
					 int w, int h, int c, 
					 int wp, int hp, int stride);
void make_patch(float* in_mat, float* out_mat, 
				  int w, int h, int c,
				  int wp, int hp, int stride);
void batch_normalization(float* in_mat, float* W, float* B, int w, int h, int c);
float* fetch_parameters(int layer_num, int block_num, int conv_num, int type, int param_ind);
float* pass_basic_block(float* in_mat, int layer_num, int block_num, 
					  int w, int h, int c, int strided, int param_ind);
void test_basic_block(int layer_num, int block_num, 
					  int w, int h, int c);
void pooling(float* in_mat, int w, int h, int c, int type);
void transpose_weight(float* in_mat, int w, int h, int c, int c_out);
void transpose_downsample(float* in_mat, int c);
void transpose_img(float* in_mat);
void enhance_dark_image(uint8_t* in_mat);