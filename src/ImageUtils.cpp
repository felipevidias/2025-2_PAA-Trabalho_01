/**
 * @file ImageUtils.cpp
 * @brief Implements utility functions for image processing and feature calculations.
 *
 * This file contains the core logic for calculating the distance between feature
 * vectors and for extracting those features (color histograms) from image files.
 */

 #include "ImageUtils.h"

 /**
  * @brief Calculates the Euclidean distance between two feature vectors.
  * @param a The first feature vector.
  * @param b The second feature vector.
  * @return The L2 norm (Euclidean distance) between vectors a and b.
  */
 float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
     float sum = 0.0;
     // Assuming vectors are of the same size, which is guaranteed by the histogram extraction.
     for (size_t i = 0; i < a.size(); i++) {
         float diff = a[i] - b[i];
         sum += diff * diff;
     }
     return sqrt(sum);
 }
 
 /**
  * @brief Extracts a color histogram from an image to serve as its feature vector.
  *
  * This function reads an image, calculates a histogram for each of the B, G, and R
  * color channels, normalizes them, and combines them into a single 1D feature vector.
  *
  * @param path The file path to the image.
  * @return A std::vector<float> of size 24 (8 bins * 3 channels) representing the
  * normalized color histogram. Returns an empty vector if the image fails to load.
  */
 std::vector<float> extractHistogram(const std::string& path) {
     // 1. Load the image from the specified path.
     cv::Mat img = cv::imread(path);
     if (img.empty()) {
         std::cerr << "Error: Could not open or find the image at: " << path << std::endl;
         return {}; // Return an empty vector on failure.
     }
 
     // 2. Split the image into its 3 color channels (B, G, R).
     std::vector<cv::Mat> bgr_planes;
     cv::split(img, bgr_planes);
 
     // 3. Define parameters for the histogram calculation.
     int histSize = 8;                // We want 8 bins per channel.
     float range[] = {0, 256};        // The pixel value range [0, 255].
     const float* histRange = {range};
     bool uniform = true, accumulate = false;
 
     // 4. Calculate the histogram for each color channel.
     cv::Mat b_hist, g_hist, r_hist;
     cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
     cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
     cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
 
     // 5. Normalize the histograms to a range of [0, 1].
     // This is crucial for a fair comparison between images of different sizes.
     cv::normalize(b_hist, b_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
     cv::normalize(g_hist, g_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
     cv::normalize(r_hist, r_hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
 
     // 6. Combine the 3 histograms into a single feature vector.
     // The values are interleaved: [B0, G0, R0, B1, G1, R1, ...].
     std::vector<float> features;
     features.reserve(histSize * 3); // Pre-allocate memory for efficiency.
     for (int i = 0; i < histSize; i++) {
         features.push_back(b_hist.at<float>(i));
         features.push_back(g_hist.at<float>(i));
         features.push_back(r_hist.at<float>(i));
     }
     
     return features;
 }