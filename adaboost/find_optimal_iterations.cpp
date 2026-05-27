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

    int lo_iterations = 1;
    int iterations = 250;
    int depth = 20;

    double bestValidationError = 1;
    int bestIteration = 0;

    cout << "Analyzing tree depths d=" << depth << '\n';
    AdaBoostClassifier adaBoostClassifier(iterations+2, depth);
    adaBoostClassifier.train(trainData);

    vector<int> it;
    vector<double> errs;

    for(int i = lo_iterations; i <= iterations; i++) {
        //double trainError = adaBoostClassifier.testAdaBoost(trainData, i);
        double validError = adaBoostClassifier.testAdaBoost(validData, i);
        //double testError = adaBoostClassifier.testAdaBoost(testData, i);
        it.push_back(i); errs.push_back(validError);
        if(validError < bestValidationError) {
            bestValidationError = validError;
            bestIteration = i;
        }
        /*cout << "Depths= " << d << " with iterations=" << i << ":\n";
        cout << "TRAIN WRONG: " << trainError << '\n';
        cout << "VALID WRONG: " << validError << '\n';
        cout << "TEST WRONG: " << testError << '\n';*/
    }

    for(auto &x : it) {
        cout << x << ", ";
    } cout << '\n';
    for(auto &x : errs) {
        cout << x << ", ";
    } cout << '\n';

    cout << "Best-performing model (based on validation set):\n";
    cout << "Best iteration: " << bestIteration << " on depth: " << depth << " with validation error " << bestValidationError << '\n';

    return 0;
}