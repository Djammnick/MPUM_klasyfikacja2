#ifndef _KNN_HPP_
#define _KNN_HPP_

#include <bits/stdc++.h>
#include "../dataio.hpp"
using namespace std;

double dist(DataPoint& L, DataPoint& R) {
    double v = 0;
    for(int u = 0; u < L.X.size(); u++) {
        v += (L.X[u]-R.X[u])*(L.X[u]-R.X[u]);
    }
    return sqrt(v);
}

class KNNClassifier {

    vector<DataPoint> data;
    int k;

public:

    KNNClassifier(vector<DataPoint>& _data, int _k) : data(_data), k(_k) {}

    void setK(int _k) {
        k = _k;
    }

    int classify(DataPoint dataPoint) {
        int target = k-1;
        
        // Sorting in O(n log n), possible to speed this up by magic fives.
        nth_element(data.begin(), data.begin()+target, data.end(), 
            [&](DataPoint& A, DataPoint& B) {return dist(A,dataPoint)<dist(B,dataPoint);});
        //sort(data.begin(), data.end(), [&](DataPoint& A, DataPoint& B) {return dist(A,dataPoint)<dist(B,dataPoint);});

        DataPoint K = data[k-1];
        double max_dist = dist(K, dataPoint);

        int ones = 0, negones = 0;
        for(int u = 0; u < k; u++) {
            (data[u].Y == 1 ? ones : negones)++;
        }

        return ((ones >= negones) ? 1 : -1);
    }

    double testKNN(vector<DataPoint>& data) {
        int correct = 0, wrong = 0;
        for(auto &datapoint : data) {
            if(classify(datapoint) == datapoint.Y) correct++;
            else wrong++;
        }
        return (double)wrong / (double)(correct+wrong);
    }
};

#endif