#include <bits/stdc++.h>
#include "svm.hpp"
#include "svm_utils.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/poly_train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    vector<DataPoint> validData = inputData("../data/poly_valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";

    vector<DataPoint> testData = inputData("../data/poly_test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoints.\n";

    SVM svm(trainData, 20000, 10);

    DataPoint weights = inputWeights("output/weights50_nomisses.txt", '\t'); // best thus far
    svm.importWeights(weights.X, weights.Y);

    testSVM(trainData, svm, "output/poly_train.txt");
    testSVM(validData, svm, "output/poly_valid.txt");
    testSVM(testData, svm, "output/poly_test.txt");

    return 0;
}