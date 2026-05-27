#ifndef _ADABOOST_HPP_
#define _ADABOOST_HPP_

#include <bits/stdc++.h>
#include "../dataio.hpp"
using namespace std;

// A binary tree of Decisions, used to classify points
// Modifications for Adaboost
class WeightedDecisionTree {
    friend class AdaBoostEstimator;

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
    vector<double> weights;

    // Structure of a problem to solve in construction of a decision tree node.
    struct DecisionProblem {
        shared_ptr<Decision>* decision;
        vector<DataPoint> dataSplit;
        vector<double> weights;
        int depth;
    };

    // Calculate the gini index of a dataset with a given parameter and its split value
    double calculateGiniIndex(vector<DataPoint>& dataSplit, vector<double>& weightsSplit, int p, int b) {
        double leftones = 0, leftnegones = 0, rightones = 0, rightnegones = 0;
        double total_weights = 0;
        for(int u = 0; u < dataSplit.size(); u++) {
            total_weights += weightsSplit[u];
            if(dataSplit[u].X[p] <= b) {
                (dataSplit[u].Y == 1 ? leftones : leftnegones) += weightsSplit[u];
            } else {
                (dataSplit[u].Y == 1 ? rightones : rightnegones) += weightsSplit[u];
            }
        }

        if(leftones + leftnegones == 0 || rightones + rightnegones == 0) return 1;

        double p_left_ones = (leftones)/(leftones+leftnegones);
        double p_left_negones = (leftnegones)/(leftones+leftnegones);
        double p_right_ones = (rightones)/(rightones+rightnegones);
        double p_right_negones = (rightnegones)/(rightones+rightnegones);

        double left_gini = 1 - p_left_ones*p_left_ones - p_left_negones*p_left_negones;
        double right_gini = 1 - p_right_ones*p_right_ones - p_right_negones*p_right_negones;

        double left_weight = (leftones+leftnegones)/total_weights;
        double right_weight = (rightones+rightnegones)/total_weights;

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
    WeightedDecisionTree(const vector<DataPoint>& _data, const vector<double>& _weights, int _depthLimit = 1, int _countLimit = 1) : data(_data), weights(_weights), depthLimit(_depthLimit), countLimit(_countLimit) {
        double ones = 0, negones = 0;
        for(int i = 0; i < data.size(); i++) {
            if(data[i].Y == 1) ones += weights[i];
            else negones += weights[i];
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
        splits.push(DecisionProblem{&root, data, weights, 0});

        while(!splits.empty()) {
            DecisionProblem curr = splits.front(); splits.pop();
            //cout << "Considering the decision problem for remaining " << curr.dataSplit.size() << " datapoints.\n";
            double ones = 0, negones = 0;
            for(int u = 0; u < curr.dataSplit.size(); u++) {
                if(curr.dataSplit[u].Y == 1) ones += curr.weights[u];
                else negones += curr.weights[u];
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
                        double gini = calculateGiniIndex(curr.dataSplit, curr.weights, p, b);
                        if(gini < bestgini) {
                            bestsplit = {p, b};
                            bestgini = gini;
                        }
                    }
                }
                *curr.decision = Decision::makeSplit(bestsplit.first, bestsplit.second);
                vector<DataPoint> left, right;
                vector<double> left_weights, right_weights;
                for(int u = 0; u < curr.dataSplit.size(); u++) {
                    if(curr.dataSplit[u].X[bestsplit.first] <= bestsplit.second) {
                        left.push_back(curr.dataSplit[u]);
                        left_weights.push_back(curr.weights[u]);
                    } else {
                        right.push_back(curr.dataSplit[u]);
                        right_weights.push_back(curr.weights[u]);
                    }
                }

                Split& mySplit = std::get<Split>((*(curr.decision))->V);
                splits.push(DecisionProblem{&mySplit.lower_or_eq, left, left_weights, curr.depth+1});
                splits.push(DecisionProblem{&mySplit.higher, right, right_weights, curr.depth+1});
            }
        }
    }

    // Export the decision structure
    void exportDecisions(string fileName) {
        ofstream fout(fileName);
        if(!fout.good()) throw runtime_error("Could not open output file at " + fileName + ".");
        serialize(root, fout);
        fout.close();
        cout << "Exported decision tree.\n";
    }
    void exportDecisions(ofstream& fout) {
        serialize(root, fout);
        cout << "Exported decision tree.\n";
    }

    // Import the decision structure
    void importDecisions(string fileName) {
        ifstream fin(fileName);
        if(!fin.good()) throw runtime_error("Could not open input file at " + fileName + ".");
        root = deserialize(fin);
        fin.close();
        cout << "Imported decision tree.\n";
    }
    void importDecisions(ifstream& fin) {
        root = deserialize(fin);
        cout << "Imported decision tree.\n";
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
    // Returns the fraction of falsely classified points.
    double testDecisionTree(const vector<DataPoint>& data) {
        int correct = 0, wrong = 0;
        for(auto &datapoint : data) {
            if(classify(datapoint) == datapoint.Y) correct++;
            else wrong++;
        }
        return (double)wrong / (double)(correct+wrong);
    }
};

struct WeakEstimator {
    shared_ptr<WeightedDecisionTree> decisionTree;
    double votePower;
};

class AdaBoostClassifier {
    vector<WeakEstimator> estimators;
    int iterations;
    int depth;

public:

    AdaBoostClassifier(int _iter, int _depth = 1) : iterations(_iter), depth(_depth) {}

    void train(vector<DataPoint>& data) {
        vector<double> weights(data.size(), 1.0/data.size());
        for(int t = 0; t < iterations; t++) {
            shared_ptr<WeightedDecisionTree> newEstimator = make_shared<WeightedDecisionTree>(WeightedDecisionTree(data, weights, depth));
            newEstimator->generateDecisions();

            // Calculating error
            double totalError = 0;
            for(int i = 0; i < data.size(); i++) {
                int pred = newEstimator->classify(data[i]);
                if(pred != data[i].Y) totalError += weights[i];
            }

            if(totalError == 0) totalError = 1.0/1e10;

            // Waga klasyfikatora
            double votingPower = log((1 - totalError) / totalError) / 2;
            double totalWeights = 0;

            // Updating the weights
            for(int i = 0; i < data.size(); i++) {
                int pred = newEstimator->classify(data[i]);
                if(pred != data[i].Y) weights[i] *= exp(votingPower);
                else weights[i] *= exp(-votingPower);
                totalWeights += weights[i];
            }

            // Normalizing weights total 1
            for(int i = 0; i < data.size(); i++) weights[i] /= totalWeights;

            estimators.push_back({newEstimator, votingPower});
        }
    }

    int classify(const DataPoint& dataPoint, int iterations = 1<<30) {
        double votes = 0;
        for(int u = 0; u < estimators.size() && u < iterations; u++) {
            votes += estimators[u].decisionTree->classify(dataPoint) * estimators[u].votePower;
        }
        return ((votes >= 0) ? 1 : -1);
    }

    double testAdaBoost(const vector<DataPoint>& data, int iterations = 1<<30) {
        int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
        for(auto &datapoint : data) {
            if(classify(datapoint, iterations) == datapoint.Y) {
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
        return (double)(falsePositive+falseNegative) / (double)(truePositive+trueNegative+falsePositive+falseNegative);
    }

    // Export the underlying structure
    void exportClassifiers(string fileName) {
        ofstream fout(fileName);
        if(!fout.good()) throw runtime_error("Could not open output file at " + fileName + ".");
        fout << estimators.size() << '\n';
        for(int u = 0; u < estimators.size(); u++) {
            estimators[u].decisionTree->exportDecisions(fout);
            fout << estimators[u].votePower << '\n';
        }
        fout.close();
        cout << "Exported AdaBoost structure.\n";
    }

    // Import the underlying structure
    void importClassifiers(string fileName) {
        ifstream fin(fileName);
        if(!fin.good()) throw runtime_error("Could not open input file at " + fileName + ".");
        int estimators_size; fin >> estimators_size;
        vector<double> empty_weights; vector<DataPoint> empty_data;
        for(int u = 0; u < estimators_size; u++) {
            WeakEstimator w;
            w.decisionTree = make_shared<WeightedDecisionTree>(WeightedDecisionTree(empty_data, empty_weights));
            w.decisionTree->importDecisions(fin);
            fin >> w.votePower;
            estimators.push_back(w);
        }
        fin.close();
        cout << "Imported AdaBoost structure.\n";
    }
};

#endif