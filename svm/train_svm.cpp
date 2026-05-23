#include <bits/stdc++.h>
#include "svm.hpp"
#include "svm_utils.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    SVM svm(trainData, 25000, 10);
    svm.calculate();

    clog << "Finished SVM training.\n";

    outputWeights(svm, "output/weights.txt");

    return 0;
}