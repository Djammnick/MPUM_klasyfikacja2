#ifndef _DATAIO_HPP_
#define _DATAIO_HPP_

// Small header for data input/output
// Simplified after the previous projects due to the usage being a nightmare previously.

#include <bits/stdc++.h>
using namespace std;

struct DataPoint {
    vector<double> X;
    double Y;
};

// Converts a string to a double value.
double toDouble(string s) {
    replace(s.begin(), s.end(), ',', '.');
    return stod(s);
}

// Converts a double value to a string.
string toReadableString(double val) {
    std::stringstream sstream;
    sstream << val;
    return sstream.str();
}

// Input data from a file with the given name.
// @optional A custom delimiter between values in one row can be given.
vector<DataPoint> inputData(const string& fileName, char delimiter = ' ') {
    ifstream fin;
    fin.open(fileName);
    if(!fin.good()) throw runtime_error("Could not open input file at " + fileName + ".");

    vector<DataPoint> data;
    DataPoint input;
    string s;

    string line;
    while(getline(fin, line)) {
        data.push_back(DataPoint());
        stringstream ss(line);
        while(getline(ss, s, delimiter)) {
            data.back().X.push_back(stoi(s));
        }
        data.back().Y = data.back().X.back();
        data.back().X.pop_back();
    }

    fin.close();

    return data;
}

DataPoint inputWeights(const string& fileName, char delimiter = ' ') {
    ifstream fin;
    fin.open(fileName);
    if(!fin.good()) throw runtime_error("Could not open input file at " + fileName + ".");

    string s;
    string line;
    getline(fin, line);
    
    DataPoint x;
    stringstream ss(line);
    while(getline(ss, s, delimiter)) {
        x.X.push_back(toDouble(s));
    } 
    x.Y = x.X.back();
    x.X.pop_back();

    fin.close();

    return x;
}

// Output data to a given file name.
// @optional A custom delimiter can be given to separate values in the same row.
void outputData(const string& fileName, vector<DataPoint>& data, char delimiter = ' ') {
    ofstream fout;
    fout.open(fileName);
    if(!fout.good()) throw runtime_error("Could not open output file at " + fileName + ".");

    for(auto &D : data) {
        for(auto &x : D.X) {
            fout << x << delimiter;
        }
        fout << D.Y << '\n';
    }
}

void outputPolyData(const string& fileName, vector<DataPoint>& data, char delimiter = ' ') {

    for(auto &D : data) {
        int S = D.X.size();
        for(int i = 0; i < S; i++) {
            D.X.push_back(D.X[i]*D.X[i]);
        }
        for(int i = 0; i < S; i++) {
            for(int j = i+1; j < S; j++) {
                D.X.push_back(D.X[i]*D.X[j]);
            }
        }
    }
    outputData(fileName, data, delimiter);
}

// Separate data randomly into three vectors with a 60-20-20 split.
// A custom seed can be given.
array<vector<DataPoint>, 3> splitData(vector<DataPoint>& data, int seed = 2026) {
    int dataSize = data.size();

    int trainSize = 0.6 * dataSize;
    int validSize = 0.2 * dataSize;
    int testSize = dataSize - trainSize - validSize;

    std::mt19937 gen(seed);
    shuffle(data.begin(), data.end(), gen);

    array<vector<DataPoint>, 3> split;
    for(int i = 0; i < trainSize; i++) split[0].push_back(data[i]);
    for(int i = 0; i < validSize; i++) split[1].push_back(data[trainSize+i]);
    for(int i = 0; i < testSize; i++) split[2].push_back(data[trainSize+validSize+i]);

    return split;
}

#endif