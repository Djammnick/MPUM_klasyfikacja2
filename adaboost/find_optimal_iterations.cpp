#include <bits/stdc++.h>
#include "adaboost.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";
    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";
    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoint.\n";

    int iterations = 100;
    int depths = 10;

    double bestValidationError = 1;
    int bestIteration = 0;
    int bestDepth = 0;

    for(int d = 1; d <= depths; d++) {
        AdaBoostClassifier adaBoostClassifier(iterations+2, d);
        adaBoostClassifier.train(trainData);

        for(int i = 1; i <= iterations; i++) {
            double trainError = adaBoostClassifier.testAdaBoost(trainData, i);
            double validError = adaBoostClassifier.testAdaBoost(validData, i);
            double testError = adaBoostClassifier.testAdaBoost(testData, i);
            if(validError < bestValidationError) {
                bestValidationError = validError;
                bestIteration = i;
                bestDepth = d;
            }
            cout << "With iterations=" << i << ":\n";
            cout << "TRAIN % WRONG: " << trainError << '\n';
            cout << "VALID % WRONG: " << validError << '\n';
            cout << "TEST % WRONG: " << testError << '\n';
        }
    }

    cout << "Best-performing model (based on validation set):\n";
    cout << "Best iteration: " << bestIteration << " on depth: " << bestDepth << " with validation error " << bestValidationError << '\n';

    return 0;
}