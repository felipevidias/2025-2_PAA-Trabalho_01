/**
 * @file ImageUtils.h
 * @brief Declares the core Document structure and utility functions for image processing.
 *
 * This header file defines the data structure used to represent an image and its
 * features. It also declares the functions for feature extraction (histogram) and
 * similarity measurement (Euclidean distance).
 */

 #ifndef IMAGE_UTILS_H
 #define IMAGE_UTILS_H
 
 // Standard Library Includes
 #include <iostream>
 #include <vector>
 #include <string>
 #include <cmath>
 #include <cfloat> // Required for FLT_MAX
 
 // Third-party Includes
 #include <opencv2/opencv.hpp> // Main header for the OpenCV library
 
 /**
  * @struct Document
  * @brief Represents a single image and its associated data within the system.
  *
  * This structure holds a unique identifier, the extracted feature vector, and
  * the original filename for reference.
  */
 struct Document {
     int id;                       ///< A unique integer identifier for the document.
     std::vector<float> features;  ///< The feature vector (e.g., color histogram).
     std::string filename;         ///< The original filename for easy identification.
 
     /**
      * @brief Default constructor.
      * Initializes a Document with a default ID of -1.
      */
     Document() : id(-1) {}
 
     /**
      * @brief Parameterized constructor.
      * @param id_ The unique ID for the document.
      * @param f The feature vector for the document.
      * @param name The original filename of the image.
      */
     Document(int id_, std::vector<float> f, std::string name = "") :
         id(id_), features(std::move(f)), filename(std::move(name)) {}
 };
 
 /**
  * @brief Calculates the Euclidean distance between two feature vectors.
  * @param a The first feature vector.
  * @param b The second feature vector.
  * @return The L2 norm (Euclidean distance) between vectors a and b.
  */
 float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b);
 
 /**
  * @brief Extracts a color histogram from an image to serve as its feature vector.
  * @param path The file path to the image.
  * @return A std::vector<float> of size 24 (8 bins * 3 channels) representing the
  * normalized color histogram. Returns an empty vector if the image fails to load.
  */
 std::vector<float> extractHistogram(const std::string& path);
 
 #endif // IMAGE_UTILS_H