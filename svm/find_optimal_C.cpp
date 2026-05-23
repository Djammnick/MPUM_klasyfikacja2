#include <bits/stdc++.h>
#include "svm.hpp"
#include "svm_utils.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    vector<double> C_values = {0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000};
    vector<SVM*> svms(C_values.size());
    for(int i = 0; i < C_values.size(); i++) {
        svms[i] = new SVM(trainData, 50000, C_values[i]);
    }

    // Quite costly calculations! Calculated weights are already stored in files.
    for(int i = 0; i < C_values.size(); i++) {
        cout << "Training SVM for C value: " << C_values[i] << '\n';
        svms[i]->calculate();
        cout << "Finished training.\n";
        outputWeights(*svms[i], "output/weights"+toReadableString(C_values[i])+".txt");
        testSVM(trainData, *svms[i], "output/train"+toReadableString(C_values[i])+".txt");
        cout << "Finished testing.\n";
    }

    return 0;
}