/**
 * @file main.cpp
 * @brief Main driver for the Algorithm Analysis project.
 * This version is configured for performance experiments with a real dataset and visualization.
 */

 #include "ImageUtils.h"     // Contains Document struct and image processing functions
 #include "DataStructures.h" // Contains the data structure implementations
 #include <chrono>           // For high-resolution timing
 
 // Helper function to display results side-by-side
 void displayResults(const std::string& queryPath, const std::string& resultPath, const std::string& methodName) {
     // Load images
     cv::Mat queryImg = cv::imread(queryPath);
     cv::Mat resultImg = cv::imread(resultPath);
 
     if (queryImg.empty() || resultImg.empty()) {
         std::cout << "Could not display images for " << methodName << std::endl;
         return;
     }
 
     // Resize images to a standard size for consistent display
     cv::Size stdSize(400, 400);
     cv::resize(queryImg, queryImg, stdSize);
     cv::resize(resultImg, resultImg, stdSize);
 
     // Create a new image to hold both side-by-side
     cv::Mat comparisonImg(stdSize.height, stdSize.width * 2, queryImg.type());
 
     // Copy query image to the left side
     queryImg.copyTo(comparisonImg(cv::Rect(0, 0, stdSize.width, stdSize.height)));
     // Copy result image to the right side
     resultImg.copyTo(comparisonImg(cv::Rect(stdSize.width, 0, stdSize.width, stdSize.height)));
     
     // Add text labels
     cv::putText(comparisonImg, "Query Image", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
     cv::putText(comparisonImg, "Found Result", cv::Point(stdSize.width + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
 
     // Show the final image in a window
     std::string windowTitle = "Result for: " + methodName;
     cv::imshow(windowTitle, comparisonImg);
 }
 
 int main() {
     //=========================================================================
     // 1. DATA CONFIGURATION AND LOADING
     //=========================================================================
     
     // --- Dataset Definition (Main Experiment) ---
     // Add more images here to test performance with larger datasets.
     std::vector<std::string> image_paths = {
         "data/db_natureza_01.jpg",
         "data/db_natureza_02.jpg",
         "data/db_natureza_03.jpg",
         "data/db_cidade_01.jpg",
         "data/db_cidade_02.jpg",
         "data/db_animal_01.jpg"
     };
 
     // --- Query Definition (Main Experiment) ---
     std::string query_path = "data/query_natureza.jpg";
 
     std::cout << ">>> STARTING PERFORMANCE EXPERIMENT <<<\n";
     std::cout << "Dataset size: " << image_paths.size() << " images." << std::endl;
     std::cout << "Query image: " << query_path << "\n" << std::endl;
 
     // --- Image Loading and Feature Extraction ---
     std::vector<Document> all_docs;
     int id = 1;
 
     for (const auto &path : image_paths) {
         std::vector<float> features = extractHistogram(path);
         if (!features.empty()) {
             all_docs.emplace_back(id++, features, path);
         }
     }
     
     // --- Prepare Query Document ---
     std::vector<float> query_features = extractHistogram(query_path);
     if (query_features.empty()) {
         std::cerr << "Failed to load the query image." << std::endl;
         return 1;
     }
     Document query(-1, query_features, query_path);
 
     const int FEATURE_DIMENSIONS = 24;
 
     //=========================================================================
     // 2. EXPERIMENTS
     //=========================================================================
 
     // --- Experiment 1: Sequential List ---
     std::cout << "--- Testing Sequential List ---" << std::endl;
     {
         DocumentList list;
         for(const auto& doc : all_docs) { list.insert(doc); }
 
         auto start_time = std::chrono::high_resolution_clock::now();
         const Document* result = list.searchSimilar(query);
         auto end_time = std::chrono::high_resolution_clock::now();
         
         if(result) {
             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
             std::cout << "Found result: " << result->filename << " (Time: " << duration.count() << " us)" << std::endl;
             // Call display function
             displayResults(query.filename, result->filename, "Sequential List");
         }
     }
 
     // --- Experiment 2: K-d Tree ---
     std::cout << "\n--- Testing K-d Tree ---" << std::endl;
     {
         KdTree tree(FEATURE_DIMENSIONS);
         for(const auto& doc : all_docs) { tree.insert(doc); }
 
         auto start_time = std::chrono::high_resolution_clock::now();
         const Document* result = tree.searchSimilar(query);
         auto end_time = std::chrono::high_resolution_clock::now();
 
         if(result) {
             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
             std::cout << "Found result: " << result->filename << " (Time: " << duration.count() << " us)" << std::endl;
             // Call display function
             displayResults(query.filename, result->filename, "K-d Tree");
         }
     }
 
     // --- Experiment 3: Locality-Sensitive Hashing (LSH) ---
     std::cout << "\n--- Testing Hashing (LSH) ---" << std::endl;
     {
         DocumentHash lsh(FEATURE_DIMENSIONS, 8, 0.25); 
         for(const auto& doc : all_docs) { lsh.insert(doc); }
 
         auto start_time = std::chrono::high_resolution_clock::now();
         const Document* result = lsh.searchSimilar(query);
         auto end_time = std::chrono::high_resolution_clock::now();
         
         if(result) {
             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
             std::cout << "Found result: " << result->filename << " (Time: " << duration.count() << " us)" << std::endl;
             // Call display function
             displayResults(query.filename, result->filename, "LSH");
         } else {
             std::cout << "No result found in the same LSH bucket." << std::endl;
         }
     }
 
     // Wait for a key press to close the image windows
     std::cout << "\nPress any key in an image window to exit." << std::endl;
     cv::waitKey(0);
     cv::destroyAllWindows();
 
     return 0;
 }