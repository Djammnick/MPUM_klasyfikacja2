#include <bits/stdc++.h>
#include "adaboost.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";
    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";
    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoints.\n";

    AdaBoostClassifier adaBoostClassifier(5);
    adaBoostClassifier.importClassifiers("output/classifiers.txt");

    double trainError = adaBoostClassifier.testAdaBoost(trainData);
    cout << "% Train Error: " << trainError << '\n';
    double validError = adaBoostClassifier.testAdaBoost(validData);
    cout << "% Valid Error: " << validError << '\n';
    double testError = adaBoostClassifier.testAdaBoost(testData);
    cout << "% Test Error: " << testError << '\n';

}