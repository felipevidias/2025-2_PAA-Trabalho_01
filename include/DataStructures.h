/**
 * @file DataStructures.h
 * @brief Declares the classes for the data structures used in the experiments.
 *
 * This header file defines the interfaces for three data structures required
 * [cite_start]by the assignment[cite: 15]:
 * 1. DocumentList: A baseline sequential list.
 * 2. KdTree: A k-dimensional tree for spatial partitioning.
 * 3. DocumentHash: An implementation of Locality-Sensitive Hashing (LSH).
 */

 #ifndef DATA_STRUCTURES_H
 #define DATA_STRUCTURES_H
 
 #include "ImageUtils.h"
 #include <vector>
 #include <map>
 #include <random>
 
 //=============================================================================
 // 1. Sequential List Structure
 //=============================================================================
 
 /**
  * @class DocumentList
  * @brief A simple data structure that stores documents in a sequential vector.
  *
  * This class serves as the baseline for performance comparison. Searches are
  * performed using a linear scan.
  */
 class DocumentList {
 private:
     std::vector<Document> docs; ///< The vector storing all documents.
 
 public:
     /**
      * @brief Inserts a new document into the list.
      * @param d The document to be added.
      */
     void insert(const Document& d);
 
     /**
      * @brief Finds the most similar document to a query via linear search.
      * @param query The query document.
      * @return A const pointer to the most similar document found.
      */
     const Document* searchSimilar(const Document& query);
 };
 
 //=============================================================================
 // 2. K-d Tree Structure
 //=============================================================================
 
 /**
  * @struct KdNode
  * @brief Represents a single node within the K-d Tree.
  */
 struct KdNode {
     Document doc;           ///< The document stored at this node.
     KdNode *left = nullptr; ///< Pointer to the left child node.
     KdNode *right = nullptr;///< Pointer to the right child node.
 
     KdNode(Document d) : doc(std::move(d)) {}
     ~KdNode() { delete left; delete right; }
 };
 
 /**
  * @class KdTree
  * @brief A k-dimensional tree for organizing points in a k-dimensional space.
  *
  * This structure enables efficient nearest neighbor searches by recursively
  * partitioning the feature space.
  */
 class KdTree {
 private:
     KdNode* root = nullptr; ///< The root node of the tree.
     int k;                  ///< The dimensionality of the feature space.
 
     /**
      * @brief Recursive helper function to insert a new node.
      */
     void insertRec(KdNode*& node, Document d, int depth);
 
     /**
      * @brief Recursive helper function to search for the nearest neighbor.
      */
     void searchSimilarRec(KdNode* node, const Document& query, const Document*& best, float& bestDist, int depth) const;
 
 public:
     /**
      * @brief Constructs a K-d Tree.
      * @param dimensions The dimensionality (k) of the feature vectors.
      */
     KdTree(int dimensions) : k(dimensions) {}
     ~KdTree() { delete root; }
 
     /**
      * @brief Inserts a new document into the tree.
      * @param d The document to be added.
      */
     void insert(const Document& d);
 
     /**
      * @brief Finds the most similar document to a query using nearest neighbor search.
      * @param query The query document.
      * @return A const pointer to the most similar document found.
      */
     const Document* searchSimilar(const Document& query);
 };
 
 //=============================================================================
 // 3. Hashing Structure (Locality-Sensitive Hashing)
 //=============================================================================
 
 /**
  * @class DocumentHash
  * @brief Implements a hash table using Locality-Sensitive Hashing (LSH).
  *
  * LSH hashes similar items into the same "bucket" with high probability,
  * allowing for fast, approximate nearest neighbor searches.
  */
 class DocumentHash {
 private:
     /// The hash table, mapping a hash key to a list of documents (a "bucket").
     std::map<std::vector<int>, std::vector<Document>> buckets;
     
     /// A set of random vectors used for projection.
     std::vector<std::vector<float>> projections;
 
     float bucketWidth; ///< A tuning parameter (w) that controls the hash bucket size.
     int numHashes;     ///< The number of hash functions to use (L).
 
     /**
      * @brief Computes the composite hash key for a feature vector.
      * @param features The feature vector to hash.
      * @return The resulting hash key.
      */
     std::vector<int> getHashKey(const std::vector<float>& features) const;
 
 public:
     /**
      * @brief Constructs the LSH table.
      * @param dimensions The dimensionality of the feature vectors.
      * @param nHashes The number of hash functions to create.
      * @param width The width of each hash bucket.
      */
     DocumentHash(int dimensions, int nHashes, float width);
 
     /**
      * @brief Inserts a new document into the hash table.
      * @param d The document to be added.
      */
     void insert(const Document& d);
 
     /**
      * @brief Finds similar documents by searching within the query's hash bucket.
      * @param query The query document.
      * @return A const pointer to the most similar document in the bucket.
      */
     const Document* searchSimilar(const Document& query);
 };
 
 
 #endif //DATA_STRUCTURES_H