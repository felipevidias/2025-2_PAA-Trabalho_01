/**
 * @file DataStructures.cpp
 * @brief Implements the data structures used for the similarity search experiments.
 * This version is updated to support Top-K nearest neighbor search.
 */

 #include "DataStructures.h"
 #include <algorithm> // for std::sort
 #include <queue>     // for std::priority_queue
 
 //=============================================================================
 // 1. DocumentList Implementation
 //=============================================================================
 
 void DocumentList::insert(const Document& d) {
     docs.push_back(d);
 }
 
 std::vector<Document> DocumentList::searchSimilar(const Document& query, int k) {
     if (docs.empty()) return {};
 
     std::vector<DocDist> distances;
     for (const auto& doc : docs) {
         float dist = euclideanDistance(query.features, doc.features);
         distances.push_back({doc, dist});
     }
 
     // Sort by distance to find the nearest neighbors
     std::sort(distances.begin(), distances.end(), [](const DocDist& a, const DocDist& b) {
         return a.dist < b.dist;
     });
 
     std::vector<Document> results;
     // Ensure we don't try to access more results than we have
     int result_count = std::min(k, (int)distances.size());
     for (int i = 0; i < result_count; ++i) {
         results.push_back(distances[i].doc);
     }
     return results;
 }
 
 
 //=============================================================================
 // 2. KdTree Implementation
 //=============================================================================
 
 void KdTree::insert(const Document& d) {
     insertRec(root, d, 0);
 }
 
 void KdTree::insertRec(KdNode*& node, Document d, int depth) {
     if (node == nullptr) {
         node = new KdNode(d);
         return;
     }
     int axis = depth % k; // FIX: Was k_dims, now matches header
     if (d.features[axis] < node->doc.features[axis]) {
         insertRec(node->left, d, depth + 1);
     } else {
         insertRec(node->right, d, depth + 1);
     }
 }
 
 std::vector<Document> KdTree::searchSimilar(const Document& query, int k) {
     if (root == nullptr) return {};
 
     std::priority_queue<DocDist> best_docs;
     
     searchSimilarRec(root, query, k, best_docs, 0);
 
     // Extract documents from the priority queue
     std::vector<Document> results;
     while (!best_docs.empty()) {
         results.push_back(best_docs.top().doc);
         best_docs.pop();
     }
     std::reverse(results.begin(), results.end()); // Reverse to get from nearest to farthest
     return results;
 }
 
 void KdTree::searchSimilarRec(KdNode* node, const Document& query, int k, std::priority_queue<DocDist>& best_docs, int depth) const {
     if (node == nullptr) return;
 
     float dist = euclideanDistance(query.features, node->doc.features);
 
     if (best_docs.size() < (size_t)k) {
         best_docs.push({node->doc, dist});
     } else if (dist < best_docs.top().dist) {
         best_docs.pop();
         best_docs.push({node->doc, dist});
     }
 
     int axis = depth % this->k; // FIX: Was k_dims, now matches header
     double diff = query.features[axis] - node->doc.features[axis];
 
     KdNode *nearChild = (diff < 0) ? node->left : node->right;
     KdNode *farChild = (diff < 0) ? node->right : node->left;
 
     searchSimilarRec(nearChild, query, k, best_docs, depth + 1);
 
     double dist_to_plane = std::abs(diff);
     if (best_docs.size() < (size_t)k || dist_to_plane < best_docs.top().dist) {
         searchSimilarRec(farChild, query, k, best_docs, depth + 1);
     }
 }
 
 
 //=============================================================================
 // 3. DocumentHash (LSH) Implementation
 //=============================================================================
 
 DocumentHash::DocumentHash(int dimensions, int nHashes, float width)
     : bucketWidth(width), numHashes(nHashes) {
     // FIX: Corrected typo from mt1997 to mt19937
     std::mt19937 gen(std::random_device{}());
     std::normal_distribution<float> dist(0.0, 1.0);
     projections.resize(numHashes);
     for (int i = 0; i < numHashes; ++i) {
         projections[i].resize(dimensions);
         for (int j = 0; j < dimensions; ++j) {
             projections[i][j] = dist(gen);
         }
     }
 }
 
 void DocumentHash::insert(const Document& d) {
     std::vector<int> key = getHashKey(d.features);
     buckets[key].push_back(d);
 }
 
 std::vector<int> DocumentHash::getHashKey(const std::vector<float>& features) const {
     std::vector<int> key;
     key.reserve(numHashes);
     for (int i = 0; i < numHashes; ++i) {
         float dotProduct = 0;
         for (size_t j = 0; j < features.size(); ++j) {
             dotProduct += features[j] * projections[i][j];
         }
         key.push_back(static_cast<int>(floor(dotProduct / bucketWidth)));
     }
     return key;
 }
 
 std::vector<Document> DocumentHash::searchSimilar(const Document& query, int k) {
     std::vector<int> queryKey = getHashKey(query.features);
     
     if (buckets.find(queryKey) == buckets.end() || buckets.at(queryKey).empty()) {
         return {};
     }
 
     std::vector<DocDist> distances;
     for (const auto& doc : buckets.at(queryKey)) {
         float dist = euclideanDistance(query.features, doc.features);
         distances.push_back({doc, dist});
     }
 
     std::sort(distances.begin(), distances.end(), [](const DocDist& a, const DocDist& b) {
         return a.dist < b.dist;
     });
 
     std::vector<Document> results;
     int result_count = std::min(k, (int)distances.size());
     for (int i = 0; i < result_count; ++i) {
         results.push_back(distances[i].doc);
     }
     return results;
 }
 
 