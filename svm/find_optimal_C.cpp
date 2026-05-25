#include <bits/stdc++.h>
#include "svm.hpp"
#include "svm_utils.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    vector<double> C_values = {20, 30, 45, 150, 200, 250};
    vector<SVM*> svms(C_values.size());
    for(int i = 0; i < C_values.size(); i++) {
        svms[i] = new SVM(trainData, 200000, C_values[i]);
    }

    // Quite costly calculations! Calculated weights are already stored in files.
    for(int i = 0; i < C_values.size(); i++) {
        cout << "Training SVM for C value: " << C_values[i] << '\n';
        svms[i]->calculate_nomisses();
        cout << "Finished training.\n";
        outputWeights(*svms[i], "output/weights"+toReadableString(C_values[i])+"_nomisses.txt");
        testSVM(trainData, *svms[i], "output/train"+toReadableString(C_values[i])+"_nomisses.txt");
        cout << "Finished testing.\n";
    }

    return 0;
}