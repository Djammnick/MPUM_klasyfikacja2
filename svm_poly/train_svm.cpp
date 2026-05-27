#include <bits/stdc++.h>
#include "svm.hpp"
#include "svm_utils.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/poly_train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    SVM svm(trainData, 50000, 50);
    svm.calculate();

    clog << "Finished SVM training.\n";

    outputWeights(svm, "output/weights.txt");

    return 0;
}