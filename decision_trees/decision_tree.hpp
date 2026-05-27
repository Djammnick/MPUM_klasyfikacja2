#ifndef _DECISION_TREE_HPP_
#define _DECISION_TREE_HPP_

#include <bits/stdc++.h>
#include "../dataio.hpp"
using namespace std;

// A binary tree of Decisions, used to classify points
class DecisionTree {

    struct Decision;
    struct Split {
        int parameter_no;
        int border;
        shared_ptr<Decision> lower_or_eq, higher;
    };
    struct Leaf {
        int classification;
    };
    struct Decision {
        variant<Leaf, Split> V;
        static shared_ptr<Decision> makeLeaf(int val) {
            shared_ptr<Decision> v = shared_ptr<Decision>(new Decision());
            v->V = Leaf{val};
            return v;
        }
        static shared_ptr<Decision> makeSplit(int parameter, int border) {
            shared_ptr<Decision> v = shared_ptr<Decision>(new Decision());
            v->V = Split{parameter, border, shared_ptr<Decision>(new Decision()), shared_ptr<Decision>(new Decision())};
            return v;
        }
    };

    shared_ptr<Decision> root;
    int depthLimit, countLimit;

    const vector<DataPoint>& data;

    // Structure of a problem to solve in construction of a decision tree node.
    struct DecisionProblem {
        shared_ptr<Decision>* decision;
        vector<DataPoint> dataSplit;
        int depth;
    };

    // Calculate the gini index of a dataset with a given parameter and its split value
    double calculateGiniIndex(vector<DataPoint>& dataSplit, int p, int b) {
        int leftones = 0, leftnegones = 0, rightones = 0, rightnegones = 0;
        for(auto &x : dataSplit) {
            if(x.X[p] <= b) {
                (x.Y == 1 ? leftones : leftnegones)++;
            } else {
                (x.Y == 1 ? rightones : rightnegones)++;
            }
        }

        if(leftones + leftnegones == 0 || rightones + rightnegones == 0) return 1;

        double p_left_ones = ((double)(leftones)/(double)(leftones+leftnegones));
        double p_left_negones = ((double)(leftnegones)/(double)(leftones+leftnegones));
        double p_right_ones = ((double)(rightones)/(double)(rightones+rightnegones));
        double p_right_negones = ((double)(rightnegones)/(double)(rightones+rightnegones));

        double left_gini = 1 - p_left_ones*p_left_ones - p_left_negones*p_left_negones;
        double right_gini = 1 - p_right_ones*p_right_ones - p_right_negones*p_right_negones;

        double left_weight = (double)(leftones+leftnegones)/(double)(dataSplit.size());
        double right_weight = (double)(rightones+rightnegones)/(double)(dataSplit.size());

        double gini = left_gini * left_weight + right_gini * right_weight;
        return gini;
    }

    // Recursively serialize the decision tree into a stream.
    void serialize(shared_ptr<Decision> node, ostream& fout) {
        if(holds_alternative<Leaf>(node->V)) {
            Leaf l = get<Leaf>(node->V);
            fout << "L " << l.classification << '\n';
        } else {
            Split s = get<Split>(node->V);
            fout << "S " << s.parameter_no << ' ' << s.border << '\n';
            serialize(s.lower_or_eq, fout);
            serialize(s.higher, fout);
        }
    }

    // Deserialize a stored decision tree.
    shared_ptr<Decision> deserialize(istream& fin) {
        string type; fin >> type;
        if(type == "L") {
            int val; fin >> val;
            return Decision::makeLeaf(val);
        } else { // type == "S"
            int p, b;
            fin >> p >> b;

            shared_ptr<Decision> node = Decision::makeSplit(p, b);
            Split& s = std::get<Split>(node->V);
            
            s.lower_or_eq = deserialize(fin);
            s.higher = deserialize(fin);

            return node;
        }
    }

public:
    // To ignore one of the requirements one can always set depthLimit=infty, countLimit=1
    DecisionTree(const vector<DataPoint>& _data, int _depthLimit, int _countLimit) : data(_data), depthLimit(_depthLimit), countLimit(_countLimit) {
        int ones = 0, negones = 0;
        for(int i = 0; i < data.size(); i++) {
            if(data[i].Y == 1) ones++;
            else negones++;
        }
        if(ones > negones) {
            root = Decision::makeLeaf(1);
        } else {
            root = Decision::makeLeaf(-1);
        }
    }

    // Create the structure of the decision tree
    void generateDecisions() {
        queue<DecisionProblem> splits;
        splits.push(DecisionProblem{&root, data, 0});

        while(!splits.empty()) {
            DecisionProblem curr = splits.front(); splits.pop();
            //cout << "Considering the decision problem for remaining " << curr.dataSplit.size() << " datapoints.\n";
            int ones = 0, negones = 0;
            for(auto &x : curr.dataSplit) {
                if(x.Y == 1) ones++;
                else negones++;
            }
            if(curr.dataSplit.size() <= countLimit || curr.depth >= depthLimit) {
                // Making a leaf
                if(ones >= negones) *curr.decision = Decision::makeLeaf(1);
                else *curr.decision = Decision::makeLeaf(-1);
            } else {
                // Making a split
                // Checking all possible splits, to determine which of them has the lowest Gini index
                pair<int, int> bestsplit = {0, 0}; // (parameter, bound)
                double bestgini = 1;
                for(int p = 0; p < curr.dataSplit[0].X.size(); p++) {
                    for(int b = -1; b < 1; b++) {
                        //cout << "Trying the split on variable " << p << " with bound <= " << b << '\n';
                        double gini = calculateGiniIndex(curr.dataSplit, p, b);
                        if(gini < bestgini) {
                            bestsplit = {p, b};
                            bestgini = gini;
                        }
                    }
                }
                *curr.decision = Decision::makeSplit(bestsplit.first, bestsplit.second);
                vector<DataPoint> left, right;
                for(auto &x : curr.dataSplit) {
                    if(x.X[bestsplit.first] <= bestsplit.second) {
                        left.push_back(x);
                    } else {
                        right.push_back(x);
                    }
                }

                Split& mySplit = std::get<Split>((*(curr.decision))->V);
                splits.push(DecisionProblem{&mySplit.lower_or_eq, left, curr.depth+1});
                splits.push(DecisionProblem{&mySplit.higher, right, curr.depth+1});
            }
        }
    }

    // Export the decision structure
    void exportDecisions(string fileName) {
        ofstream fout(fileName);
        if(!fout.good()) throw runtime_error("Could not open output file at " + fileName + ".");
        serialize(root, fout);
        fout.close();
        //cout << "Exported decision tree.\n";
    }

    // Import the decision structure
    void importDecisions(string fileName) {
        ifstream fin(fileName);
        if(!fin.good()) throw runtime_error("Could not open input file at " + fileName + ".");
        root = deserialize(fin);
        fin.close();
        //cout << "Imported decision tree.\n";
    }

    // Classifies a single point.
    int classify(const DataPoint& dataPoint) {
        shared_ptr<Decision> curr = root;
        while(true) {
            if(holds_alternative<Leaf>(curr->V)) {
                return get<Leaf>(curr->V).classification;
            } else {
                Split s = get<Split>(curr->V);
                if(dataPoint.X[s.parameter_no] <= s.border) curr = s.lower_or_eq;
                else curr = s.higher;
            }
        }
    }

    // Test the decision tree on a given set of datapoints.
    // Returns the percentage of falsely classified points.
    /*double testDecisionTree(const vector<DataPoint>& data) {
        int correct = 0, wrong = 0;
        for(auto &datapoint : data) {
            if(datapoint.Y == classify(datapoint)) correct++;
            else wrong++;
        }
        return (double)wrong / (double)(correct+wrong);
    }*/

    double testDecisionTree(const vector<DataPoint>& data) {
        int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
        for(auto &datapoint : data) {
            if(classify(datapoint) == datapoint.Y) {
                if(datapoint.Y == 1) truePositive++;
                else trueNegative++;
            }
            else {
                if(datapoint.Y == 1) falsePositive++;
                else falseNegative++;
            }
        }
        cout << "Dokladnosc: " << (double)(truePositive+trueNegative) / (double)(truePositive+trueNegative+falsePositive+falseNegative) << '\n';
        cout << "Precyzja: " << (double)(truePositive) / (double)(truePositive+falseNegative) << '\n';
        return (double)(truePositive+trueNegative) / (double)(truePositive+trueNegative+falsePositive+falseNegative);
    
    }
};

#endif