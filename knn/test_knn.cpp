#include <bits/stdc++.h>
#include "knn.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";
    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";
    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoints.\n";

    KNNClassifier knn(trainData, 5);

    double trainError = knn.testKNN(trainData);
    cout << "% Train Error: " << trainError << '\n';
    double validError = knn.testKNN(validData);
    cout << "% Valid Error: " << validError << '\n';
    double testError = knn.testKNN(testData);
    cout << "% Test Error: " << testError << '\n';
}