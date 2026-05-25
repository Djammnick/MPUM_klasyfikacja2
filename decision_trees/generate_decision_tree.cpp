#include <bits/stdc++.h>
#include "decision_tree.hpp"

using namespace std;

int main() {
    vector<DataPoint> trainData = inputData("../data/train.data", '\t');
    clog << "Successfully read " << trainData.size() << " training datapoints.\n";

    DecisionTree decisionTree(trainData, 5, 10);
    decisionTree.generateDecisions();

    decisionTree.exportDecisions("output/decision_tree.txt");
}