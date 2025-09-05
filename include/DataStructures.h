/**
 * @file DataStructures.h
 * @brief Declares the classes for the data structures used in the experiments.
 */

 #ifndef DATA_STRUCTURES_H
 #define DATA_STRUCTURES_H
 
 #include "ImageUtils.h"
 #include <vector>
 #include <map>
 #include <random>
 #include <queue> // Required for priority_queue
 
 //=============================================================================
 // Helper Structure for KNN Search
 //=============================================================================
 
 /**
  * @struct DocDist
  * @brief A helper struct to pair a Document with its distance to a query.
  * This is used for sorting and in priority queues to find the K-nearest neighbors.
  */
 struct DocDist {
     Document doc;
     float dist;
 
     // Overload the less-than operator to make this struct work in a max-heap
     // (std::priority_queue). The item with the LARGEST distance will be at the top.
     bool operator<(const DocDist& other) const {
         return dist < other.dist;
     }
 };
 
 
 //=============================================================================
 // 1. Sequential List Structure
 //=============================================================================
 class DocumentList {
 private:
     std::vector<Document> docs;
 
 public:
     void insert(const Document& d);
     std::vector<Document> searchSimilar(const Document& query, int k);
 };
 
 //=============================================================================
 // 2. K-d Tree Structure
 //=============================================================================
 struct KdNode {
     Document doc;
     KdNode *left = nullptr;
     KdNode *right = nullptr;
 
     KdNode(Document d) : doc(std::move(d)) {}
     ~KdNode() { delete left; delete right; }
 };
 
 class KdTree {
 private:
     KdNode* root = nullptr;
     int k; // The dimensionality of the feature space.
 
     void insertRec(KdNode*& node, Document d, int depth);
     void searchSimilarRec(KdNode* node, const Document& query, int k, std::priority_queue<DocDist>& best_docs, int depth) const;
 
 public:
     KdTree(int dimensions) : k(dimensions) {}
     ~KdTree() { delete root; }
 
     void insert(const Document& d);
     std::vector<Document> searchSimilar(const Document& query, int k);
 };
 
 //=============================================================================
 // 3. Hashing Structure (Locality-Sensitive Hashing)
 //=============================================================================
 class DocumentHash {
 private:
     std::map<std::vector<int>, std::vector<Document>> buckets;
     std::vector<std::vector<float>> projections;
     float bucketWidth;
     int numHashes;
 
     std::vector<int> getHashKey(const std::vector<float>& features) const;
 
 public:
     DocumentHash(int dimensions, int nHashes, float width);
     void insert(const Document& d);
     std::vector<Document> searchSimilar(const Document& query, int k);
 };
 
 #endif //DATA_STRUCTURES_H
 
 