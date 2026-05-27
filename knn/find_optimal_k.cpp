#include <bits/stdc++.h>
#include "knn.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";
    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";
    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoint.\n";

    vector<int> ks;
    vector<double> errs; 
    int k_limit = 30;
    KNNClassifier knn(trainData, 0);
    double bestValidationError = 1;
    int bestK = 0;
    for(int i = 1; i <= k_limit; i++) {
        knn.setK(i);
        double trainError = knn.testKNN(trainData);
        double validError = knn.testKNN(validData);
        double testError = knn.testKNN(testData);
        if(validError < bestValidationError) {
            bestValidationError = validError;
            bestK = i;
        }
        //cout << "With k=" << i << ":\n";
        //cout << "TRAIN % WRONG: " << trainError << '\n';
        //cout << "VALID % WRONG: " << validError << '\n';
        //cout << "TEST % WRONG: " << testError << '\n';
        ks.push_back(i);
        errs.push_back(validError);
    }

    for(auto dk : ks) {
        cout << dk << ' ';
    } cout << '\n';
    for(auto derr : errs) {
        cout << derr << ' ';
    } cout << '\n';

    cout << "Best-performing model (based on validation set):\n";
    cout << "Best iteration: " << bestK << " with validation error " << bestValidationError << '\n';

    return 0;
}