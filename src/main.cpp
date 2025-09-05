/**
 * @file main.cpp
 * @brief Main driver for the Algorithm Analysis project.
 * This version is adapted for flat-directory datasets where categories are
 * determined by filename ranges (e.g., Wang Database).
 */

 #include "ImageUtils.h"
 #include "DataStructures.h"
 #include <chrono>
 #include <filesystem>
 #include <fstream>
 #include <algorithm>
 #include <vector>
 
 namespace fs = std::filesystem;
 
 // Helper function to get the category from a filename (e.g., "data/150.jpg" -> category 1)
 // It assumes images are grouped in hundreds.
 int getCategory(const std::string& filename) {
     try {
         fs::path p(filename);
         std::string stem = p.stem().string(); // Gets the filename without extension (e.g., "150")
         int id = std::stoi(stem);
         return id / 100; // Group images by hundreds
     } catch (...) {
         return -1; // Return an invalid category if filename is not a number
     }
 }
 
 int main() {
     //=========================================================================
     // 1. DATA CONFIGURATION AND LOADING
     //=========================================================================
     
     // --- Automatically load all image paths from the "data" directory ---
     std::vector<std::string> image_paths;
     const std::string data_path = "data";
     for (const auto & entry : fs::directory_iterator(data_path)) {
         if (entry.is_regular_file()) {
             std::string extension = entry.path().extension().string();
             if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                 image_paths.push_back(entry.path().string());
             }
         }
     }
 
     if (image_paths.empty()) {
         std::cerr << "Error: No images found in the 'data' directory." << std::endl;
         return 1;
     }
 
     // --- Query Definitions ---
     // Select one image from each of a few categories for robust testing.
     // IMPORTANT: Make sure these files exist in your 'data' folder.
     std::vector<std::string> query_paths = {
         "data/50.jpg",   // Categoria 0 (e.g., Africa)
         "data/150.jpg",  // Categoria 1 (e.g., Praia)
         "data/250.jpg",  // Categoria 2 (e.g., Monumentos)
         "data/450.jpg",  // Categoria 4 (e.g., Flores)
         "data/650.jpg",  // Categoria 6 (e.g., Cavalos)
         "data/950.jpg"   // Categoria 9 (e.g., Comida)
     };
 
     // --- Prepare results file ---
     std::ofstream resultsFile("results.txt");
     if (!resultsFile.is_open()) {
         std::cerr << "Error: Could not open results.txt for writing." << std::endl;
         return 1;
     }
 
     std::cout << "Starting experiments with large dataset... This may take a while." << std::endl;
     std::cout << "Results will be saved to results.txt" << std::endl;
     resultsFile << "PERFORMANCE AND PRECISION ANALYSIS (Flat Directory Dataset)\n";
     resultsFile << "================================================================\n";
     resultsFile << "Total images in database: " << image_paths.size() << "\n\n";
 
     // --- Load all documents into memory once to be fair in timing ---
     std::cout << "Loading and extracting features from " << image_paths.size() << " images..." << std::endl;
     std::vector<Document> all_docs;
     int id_counter = 1;
     for (const auto &path : image_paths) {
         std::vector<float> features = extractHistogram(path);
         if (!features.empty()) {
             all_docs.emplace_back(id_counter++, features, path);
         }
     }
     std::cout << "Feature extraction complete.\n" << std::endl;
 
     const int FEATURE_DIMENSIONS = 24;
     const int TOP_K = 10;
 
     //=========================================================================
     // 2. EXPERIMENTS LOOP
     //=========================================================================
     for (const auto& query_path : query_paths) {
         Document query;
         bool query_found = false;
         for(const auto& doc : all_docs){
             if(doc.filename == query_path){
                 query = doc;
                 query_found = true;
                 break;
             }
         }
         if(!query_found){
             std::cerr << "Warning: Query image " << query_path << " not found in the dataset. Skipping." << std::endl;
             continue;
         }
 
         int queryCategory = getCategory(query.filename);
         resultsFile << "--------------------------------------\n";
         resultsFile << "QUERY IMAGE: " << query.filename << " (Category " << queryCategory << ")\n";
         resultsFile << "--------------------------------------\n\n";
         
         // --- Experiment 1: Sequential List ---
         {
             DocumentList list;
             for(const auto& doc : all_docs) { if(doc.filename != query.filename) list.insert(doc); }
             
             auto start_time = std::chrono::high_resolution_clock::now();
             std::vector<Document> results = list.searchSimilar(query, TOP_K);
             auto end_time = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
 
             int correct_count = 0;
             resultsFile << "--- Method: Sequential List ---\n";
             resultsFile << "Time: " << duration.count() << " ms\n";
             for(const auto& res : results){
                 if(getCategory(res.filename) == queryCategory) correct_count++;
             }
             double precision = (double)correct_count / TOP_K * 100.0;
             resultsFile << "Precision@" << TOP_K << ": " << precision << "%\n\n";
         }
 
         // --- Experiment 2: K-d Tree ---
         {
             KdTree tree(FEATURE_DIMENSIONS);
             for(const auto& doc : all_docs) { if(doc.filename != query.filename) tree.insert(doc); }
 
             auto start_time = std::chrono::high_resolution_clock::now();
             std::vector<Document> results = tree.searchSimilar(query, TOP_K);
             auto end_time = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
             
             int correct_count = 0;
             resultsFile << "--- Method: K-d Tree ---\n";
             resultsFile << "Time: " << duration.count() << " us\n";
             for(const auto& res : results){
                 if(getCategory(res.filename) == queryCategory) correct_count++;
             }
             double precision = (double)correct_count / TOP_K * 100.0;
             resultsFile << "Precision@" << TOP_K << ": " << precision << "%\n\n";
         }
 
         // --- Experiment 3: Locality-Sensitive Hashing (LSH) ---
         {
             DocumentHash lsh(FEATURE_DIMENSIONS, 16, 0.5); 
             for(const auto& doc : all_docs) { if(doc.filename != query.filename) lsh.insert(doc); }
 
             auto start_time = std::chrono::high_resolution_clock::now();
             std::vector<Document> results = lsh.searchSimilar(query, TOP_K);
             auto end_time = std::chrono::high_resolution_clock::now();
             auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
 
             int correct_count = 0;
             resultsFile << "--- Method: Hashing (LSH) ---\n";
             resultsFile << "Time: " << duration.count() << " us\n";
             if(results.empty()){
                 resultsFile << "No results found in the same LSH bucket.\n";
                 resultsFile << "Precision@" << TOP_K << ": 0.0%\n\n";
             } else {
                 for(const auto& res : results){
                     if(getCategory(res.filename) == queryCategory) correct_count++;
                 }
                 double precision = (double)correct_count / results.size() * 100.0;
                 resultsFile << "Precision@" << results.size() << " (on returned items): " << precision << "%\n\n";
             }
         }
     }
 
     resultsFile.close();
     std::cout << "\nExperiments finished successfully. Check results.txt for the output." << std::endl;
     return 0;
 }
 
 