#ifndef _SVM_UTILS_HPP_
#define _SVM_UTILS_HPP_

#include <bits/stdc++.h>
#include "svm.hpp"

using namespace std;

void outputWeights(SVM& svm, string outputFile) {
    optional<pair<vector<double>, double>> V = svm.exportWeights();
    if(!V.has_value()) {
        cout << "Could not output weights - the SVM failed the calculation.\n";
        return;
    }
    ofstream fout; fout.open(outputFile);
    if(!fout.good()) throw runtime_error("Could not open training data output file.");

    for(auto &d : V.value().first) {
        fout << d << '\t';
    }
    fout << V.value().second << '\n';
}

void testSVM(vector<DataPoint>& data, SVM& svm, string outputFile) {
    cout << "Writing data output to " << outputFile << '\n';
    ofstream fout; fout.open(outputFile);
    int cnt = 0;
    double ev;
    if(!fout.good()) throw runtime_error("Could not open training data output file.");
    for(int i = 0; i < data.size(); i++) {
        ev = svm.evaluate(data[i]);
        fout << toReadableString(ev) << '\t';
        if(ev <= 0) { fout << "WRONG\n"; cnt++; }
        else if(ev < 1) fout << "PREMARGIN\n";
        else if(ev == 1) fout << "MARGIN\n";
        else fout << "OK\n";
    }
    cout << "% WRONG: " << (double)cnt/data.size() << '\n';
    fout.close();
}



#endif