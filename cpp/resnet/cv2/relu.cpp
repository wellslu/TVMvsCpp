#include "modules.hpp"

cv::Mat ReLU::forward(const cv::Mat &input)
{
    // Create an output matrix with the same size and type as input
    cv::Mat output = input.clone(); // Clone input to create a new matrix for output

    // Apply ReLU: Set all negative values to 0
    cv::max(output, 0, output); // Element-wise max: ReLU(x) = max(0, x)

    return output;
}