#include <bits/stdc++.h>
#include "adaboost.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    AdaBoostClassifier adaBoostClassifier(5);
    adaBoostClassifier.train(trainData);

    double trainError = adaBoostClassifier.testAdaBoost(trainData);
    cout << "TRAIN ERROR %: " << trainError << '\n';

    adaBoostClassifier.exportClassifiers("output/classifiers.txt");
}