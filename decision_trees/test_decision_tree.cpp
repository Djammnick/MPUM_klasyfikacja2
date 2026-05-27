#include <bits/stdc++.h>
#include "decision_tree.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";
    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";
    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoints.\n";

    DecisionTree decisionTree(trainData, 19, 1);
    decisionTree.importDecisions("output/decision_tree1_19.txt");
    
    double trainError = decisionTree.testDecisionTree(trainData);
    cout << "Train Error: " << trainError << '\n';
    double validError = decisionTree.testDecisionTree(validData);
    cout << "Valid Error: " << validError << '\n';
    double testError = decisionTree.testDecisionTree(testData);
    cout << "Test Error: " << testError << '\n';
}