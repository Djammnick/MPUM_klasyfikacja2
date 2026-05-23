#include <bits/stdc++.h>
#include "svm.hpp"
#include "svm_utils.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";

    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoints.\n";

    SVM svm(trainData, 20000, 10);

    DataPoint weights = inputWeights("output/weights.txt", '\t');
    svm.importWeights(weights.X, weights.Y);

    testSVM(trainData, svm, "output/train.txt");
    testSVM(validData, svm, "output/valid.txt");
    testSVM(testData, svm, "output/test.txt");

    return 0;
}