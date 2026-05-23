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

    vector<double> C_values = {0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000};
    vector<SVM*> svms(C_values.size());
    for(int i = 0; i < C_values.size(); i++) {
        svms[i] = new SVM(trainData, 50000, C_values[i]);
        DataPoint weights = inputWeights("output/weights"+toReadableString(C_values[i])+".txt", '\t');
        svms[i]->importWeights(weights.X, weights.Y);
        cout << "Testing SVM with C = " << C_values[i] << '\n';
        testSVM(trainData, *svms[i], "output/train"+toReadableString(C_values[i])+".txt");
        testSVM(validData, *svms[i], "output/valid"+toReadableString(C_values[i])+".txt");
        testSVM(testData, *svms[i], "output/test"+toReadableString(C_values[i])+".txt");
    }

    return 0;
}