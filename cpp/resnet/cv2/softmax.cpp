#include "modules.hpp"

cv::Mat SoftmaxLayer::forward(const cv::Mat &input)
{
    // Ensure the input is in float format
    cv::Mat inputFloat;
    input.convertTo(inputFloat, CV_32F);

    // Get the number of rows (samples) and columns (features)
    int numRows = inputFloat.rows;

    // Output matrix to store softmax results
    cv::Mat output = cv::Mat::zeros(inputFloat.size(), inputFloat.type());

    for (int i = 0; i < numRows; ++i)
    {
        // Extract one row (which represents one sample)
        cv::Mat row = inputFloat.row(i);

        // Subtract the maximum value in the row for numerical stability
        double maxVal;
        cv::minMaxLoc(row, nullptr, &maxVal);
        row = row - maxVal;

        // Calculate the exponentials of each element
        cv::Mat expRow;
        cv::exp(row, expRow);

        // Sum of exponentials
        cv::Scalar sumExp = cv::sum(expRow);

        // Normalize the row by dividing each element by the sum of exponentials
        expRow = expRow / sumExp[0];

        // Store the result in the output matrix
        expRow.copyTo(output.row(i));
    }

    return output;
}