//
// Created by Moritz on 19.04.2021.
//

#ifndef FASTMARCHING2_HEAPANDSTRUCTVECTOR_H
#define FASTMARCHING2_HEAPANDSTRUCTVECTOR_H


#include<cfloat>
#include <string>
#include "settings.h"
#include <limits>
#include <vector>
//A simple representation of the points with weight
struct WeightedPoint
{

    short m_x{0};
    short m_y{0};
    short m_z{0};
    //std::string status{"Far"};
    double weight{std::numeric_limits<double>::infinity()};
    /* WeightedPoint()
     {
         m_x, m_y, m_z=0,0,0;
         weight = std::numeric_limits<double>::infinity();
     }
     WeightedPoint(const WeightedPoint &weightedPoint)
     {
         m_x, m_y, m_z=0,0,0;
         weight = std::numeric_limits<double>::infinity();
     }*/

};



// A class for Min Heap
class MinHeap
{
    std::vector<WeightedPoint> harr; // pointer to array of elements in heap
    int* table; // pointer to table of indices for a given element (table[x+y*x_grid_size+z*x_grid_size*y_grid_size]=harr[(x,y,z)] if in heap, -1 else
    int capacity; // maximum possible size of min heap
    int heap_size; // Current number of elements in min heap
public:
    // Constructor
    MinHeap(int capacity);

    //Destructor
    ~MinHeap();

    //returns size of heap
    int get_size() {return heap_size;}

    // to heapify a subtree with the root at given index
    void MinHeapify(int );

    static int parent(int i) { return (i-1)/2; }

    // to get index of left child of node at index i
    static int left(int i) { return (2*i + 1); }

    // to get index of right child of node at index i
    static int right(int i) { return (2*i + 2); }

    //returns the index of element (x,y,z)
    int get_heap_index(short x, short y, short z) const;

    //returns the weight at idex i
    double weight_at_index(const int i);

    // to extract the root which is the minimum element
    WeightedPoint extractMin();

    // Decreases key value of key at index i to new_val
    void decreaseKey(int i, double new_val);

    // Returns the minimum key (key at root) from min heap
    WeightedPoint getMin() { return harr[0]; }

    // Deletes a key stored at index i
    void deleteKey(int i);

    // Inserts a new key 'k'
    void insertKey(WeightedPoint k);

    // Prototype of a utility function to swap two points
    void swap(WeightedPoint *x, WeightedPoint *y);
};

#endif //FASTMARCHING2_HEAPANDSTRUCTVECTOR_H
