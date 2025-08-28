/**
 * @file DataStructures.cpp
 * @brief Implements the data structures used for the similarity search experiments.
 *
 * This file contains the method implementations for the following classes:
 * 1. DocumentList: A simple sequential list with linear search.
 * 2. KdTree: A k-dimensional tree for efficient spatial searching.
 * 3. DocumentHash: A hash table using Locality-Sensitive Hashing (LSH).
 */

 #include "DataStructures.h"

 //=============================================================================
 // 1. DocumentList Implementation
 //=============================================================================
 
 /**
  * @brief Inserts a document into the list.
  * @param d The Document object to insert.
  * @note This is an amortized O(1) operation due to std::vector's behavior.
  */
 void DocumentList::insert(const Document& d) {
     docs.push_back(d);
 }
 
 /**
  * @brief Finds the most similar document to a query using linear search.
  * @param query The document to find neighbors for.
  * @return A const pointer to the most similar document in the list. Returns
  * nullptr if the list is empty.
  * @note This is an O(N * D) operation, where N is the number of documents and
  * D is the dimensionality of the feature vectors.
  */
 const Document* DocumentList::searchSimilar(const Document& query) {
     if (docs.empty()) return nullptr;
     
     const Document* best = nullptr;
     float bestDist = FLT_MAX;
 
     // Iterate through every document and calculate the distance.
     for (const auto &d : docs) {
         float dist = euclideanDistance(query.features, d.features);
         if (dist < bestDist) {
             bestDist = dist;
             best = &d;
         }
     }
     return best;
 }
 
 
 //=============================================================================
 // 2. KdTree Implementation
 //=============================================================================
 
 /**
  * @brief Public entry point to insert a document into the K-d Tree.
  * @param d The document to insert.
  */
 void KdTree::insert(const Document& d) {
     insertRec(root, d, 0);
 }
 
 /**
  * @brief Recursively finds the correct position and inserts a new node.
  * @param node The current node in the recursion.
  * @param d The document to insert.
  * @param depth The current depth in the tree, used to determine the axis.
  */
 void KdTree::insertRec(KdNode*& node, Document d, int depth) {
     // Base case: If the current node is null, we've found the insertion point.
     if (node == nullptr) {
         node = new KdNode(d);
         return;
     }
 
     // Determine the axis to split on (cycles through 0, 1, 2, ..., k-1).
     int axis = depth % k;
 
     // Recursive step: Decide whether to go down the left or right subtree.
     if (d.features[axis] < node->doc.features[axis]) {
         insertRec(node->left, d, depth + 1);
     } else {
         insertRec(node->right, d, depth + 1);
     }
 }
 
 /**
  * @brief Public entry point for finding the nearest neighbor to a query document.
  * @param query The document to find the nearest neighbor for.
  * @return A const pointer to the most similar document. Returns nullptr if empty.
  */
 const Document* KdTree::searchSimilar(const Document& query) {
     if (root == nullptr) return nullptr;
 
     const Document* best = nullptr;
     float bestDist = FLT_MAX;
     searchSimilarRec(root, query, best, bestDist, 0);
     return best;
 }
 
 /**
  * @brief Recursively searches the tree for the nearest neighbor.
  * @param node The current node in the recursion.
  * @param query The query document.
  * @param best A reference to the pointer of the best-so-far document.
  * @param bestDist A reference to the smallest distance found so far.
  * @param depth The current depth in the tree.
  */
 void KdTree::searchSimilarRec(KdNode* node, const Document& query, const Document*& best, float& bestDist, int depth) const {
     if (node == nullptr) return;
 
     // Check the distance from the query to the current node.
     float dist = euclideanDistance(query.features, node->doc.features);
     if (dist < bestDist) {
         bestDist = dist;
         best = &node->doc;
     }
 
     // Determine the splitting axis for the current depth.
     int axis = depth % k;
     float diff = query.features[axis] - node->doc.features[axis];
 
     // Determine which subtree is "near" (contains the query point) and which is "far".
     KdNode *nearChild = (diff < 0) ? node->left : node->right;
     KdNode *farChild = (diff < 0) ? node->right : node->left;
 
     // Recursively search the "near" subtree first.
     searchSimilarRec(nearChild, query, best, bestDist, depth + 1);
 
     // Pruning Step: Only search the "far" subtree if it's possible it could
     // contain a point closer than the current best distance. This is checked by
     // comparing the distance to the splitting plane with the best distance.
     if (std::abs(diff) < bestDist) {
         searchSimilarRec(farChild, query, best, bestDist, depth + 1);
     }
 }
 
 
 //=============================================================================
 // 3. DocumentHash (LSH) Implementation
 //=============================================================================
 
 /**
  * @brief Constructs the LSH hash table.
  * @param dimensions The dimensionality of the feature vectors (k).
  * @param nHashes The number of hash functions to use (L).
  * @param width The width of the buckets (w), a key tuning parameter.
  */
 DocumentHash::DocumentHash(int dimensions, int nHashes, float width)
     : bucketWidth(width), numHashes(nHashes) {
     
     // Set up a random number generator with a normal distribution.
     std::mt19937 gen(std::random_device{}());
     std::normal_distribution<float> dist(0.0, 1.0);
 
     // Create 'nHashes' random projection vectors, each with 'dimensions' elements.
     projections.resize(numHashes);
     for (int i = 0; i < numHashes; ++i) {
         projections[i].resize(dimensions);
         for (int j = 0; j < dimensions; ++j) {
             projections[i][j] = dist(gen);
         }
     }
 }
 
 /**
  * @brief Inserts a document into the LSH hash table.
  * @param d The document to insert.
  */
 void DocumentHash::insert(const Document& d) {
     // Calculate the hash key and add the document to the corresponding bucket.
     std::vector<int> key = getHashKey(d.features);
     buckets[key].push_back(d);
 }
 
 /**
  * @brief Computes the LSH hash key for a given feature vector.
  * @param features The feature vector to hash.
  * @return A vector of integers representing the composite hash key.
  */
 std::vector<int> DocumentHash::getHashKey(const std::vector<float>& features) const {
     std::vector<int> key;
     key.reserve(numHashes);
 
     // For each hash function (i.e., each random projection vector)...
     for (int i = 0; i < numHashes; ++i) {
         // ...calculate the dot product between features and the projection vector...
         float dotProduct = 0;
         for (size_t j = 0; j < features.size(); ++j) {
             dotProduct += features[j] * projections[i][j];
         }
         // ...then discretize the result to get a bucket index.
         key.push_back(static_cast<int>(floor(dotProduct / bucketWidth)));
     }
     return key;
 }
 
 /**
  * @brief Finds the most similar document by searching only within one bucket.
  * @param query The query document.
  * @return A const pointer to the most similar document in the query's bucket.
  * Returns nullptr if the bucket is empty.
  */
 const Document* DocumentHash::searchSimilar(const Document& query) {
     // Calculate the hash key for the query document.
     std::vector<int> queryKey = getHashKey(query.features);
     
     // Check if the corresponding bucket exists and is not empty.
     if (buckets.find(queryKey) == buckets.end() || buckets.at(queryKey).empty()) {
         // Note: A more robust implementation might also search neighboring buckets.
         return nullptr;
     }
 
     // Perform a linear scan only on the documents within this single bucket.
     const Document* best = nullptr;
     float bestDist = FLT_MAX;
     for (const auto &d : buckets.at(queryKey)) {
         float dist = euclideanDistance(query.features, d.features);
         if (dist < bestDist) {
             bestDist = dist;
             best = &d;
         }
     }
     return best;
 }