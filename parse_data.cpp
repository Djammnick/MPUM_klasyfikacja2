#include "dataio.hpp"

int main() {
    vector<DataPoint> data = inputData("phishing.data", ',');

    array<vector<DataPoint>, 3> split = splitData(data);

    outputData("data/train.data", split[0], '\t');
    outputData("data/valid.data", split[1], '\t');
    outputData("data/test.data", split[2], '\t');

    outputPolyData("data/poly_train.data", split[0], '\t');
    outputPolyData("data/poly_valid.data", split[1], '\t');
    outputPolyData("data/poly_test.data", split[2], '\t');
}