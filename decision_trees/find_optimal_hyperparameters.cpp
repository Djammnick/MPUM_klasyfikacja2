#include <bits/stdc++.h>
#include "decision_tree.hpp"
#include "../dataio.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";
    vector<DataPoint> validData = inputData("../data/valid.data", '\t');
    clog << "Successfully read " << validData.size() << " validation datapoints.\n";
    vector<DataPoint> testData = inputData("../data/test.data", '\t');
    clog << "Successfully read " << testData.size() << " testing datapoint.\n";

    vector<int> split_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 40};
    vector<int> depth_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37, 40};

    vector<double> splits;
    vector<double> depths;
    vector<double> valids;

    double bestValidationError = 1;
    pair<int, int> bestPair;
    vector<DecisionTree*> decisionTrees(split_values.size() * depth_values.size());
    for(int i = 0; i < split_values.size(); i++) {
        for(int j = 0; j < depth_values.size(); j++) {
            DecisionTree decisionTree(trainData, depth_values[j], split_values[i]);
            decisionTree.generateDecisions();
            decisionTree.exportDecisions("output/decision_tree"+toReadableString(split_values[i])+"_"+toReadableString(depth_values[j])+".txt");
            double trainError = decisionTree.testDecisionTree(trainData);
            double validError = decisionTree.testDecisionTree(validData);
            double testError = decisionTree.testDecisionTree(testData);
            if(validError < bestValidationError) {
                bestValidationError = validError;
                bestPair = {depth_values[j], split_values[i]};
            }
            splits.push_back(split_values[i]);
            depths.push_back(depth_values[j]);
            valids.push_back(validError);
            //cout << "With split_limit=" << split_values[i] << " and depth_limit=" << depth_values[j] << "\n";
            //cout << "TRAIN % WRONG: " << trainError << '\n';
            //cout << "VALID % WRONG: " << validError << '\n';
            //cout << "TEST % WRONG: " << testError << '\n';
        }
    }

    for(auto &s : splits) cout << s << ", ";
    cout << '\n';
    for(auto &d : depths) cout << d << ", ";
    cout << '\n';
    for(auto &e : valids) cout << e << ", ";
    cout << '\n';

    cout << "Best-performing model (based on validation set):\n";
    cout << "Depth limit: " << bestPair.first << ' ' << "Split limit: " << bestPair.second << " with validation error " << bestValidationError << '\n';

    return 0;
}