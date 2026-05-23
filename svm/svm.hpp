#ifndef _SVM_HPP_
#define _SVM_HPP_

#include <bits/stdc++.h>
#include "../dataio.hpp"
#include "../random.hpp"

using namespace std;

class SVM {
    const vector<DataPoint>& data;
    int epochs;
    vector<double> w;
    double b;
    int m, param;
    double C;
    bool found = false;

    double dotprod(const vector<double>& L, const vector<double>& R) {
        double res = 0;
        for(int u = 0; u < param; u++) {
            res += L[u]*R[u];
        }
        return res;
    }

    unordered_map<long long, double> memorized;
    // Calculates the dot product of data points i and j.
    // Added memorization of previous values to cut the calculation time.
    double dotprod(int i, int j) {
        long long ip = i, jp = j;
        if(memorized.count((ip<<32)+jp) > 0) return memorized[(ip<<32)+jp];
        if(memorized.count((jp<<32)+ip) > 0) return memorized[(jp<<32)+ip];
        double val = dotprod(data[i].X, data[j].X);
        memorized[(jp<<32)+ip] = val;
        return val;
    }

    double eval(vector<double>& alpha) {
        // Evaluate the value of f with the given alpha coefficients.
        double term1 = 0;
        for(int u = 0; u < m; u++) {
            term1 += alpha[u];
        }

        double term2 = 0;
        for(int i = 0; i < m; i++) {
            if(alpha[i] == 0) continue;
            for(int j = 0; j < m; j++) {
                if(alpha[j] == 0) continue;
                double dprod = dotprod(i, j);
                term2 += alpha[i]*alpha[j]*data[i].Y*data[j].Y*dprod;
            }
        }
        return term1-(0.5*term2);
    }

    pair<double, double> intersect(int i, int j, double zeta) {
        // Calculates L and H - restrictions for the value of \alpha_j
        double C_m = C/m;
        double L, H;

        double left_int = zeta / data[j].Y; // Intersection with left edge
        double right_int = (zeta - (C_m * data[i].Y)) / data[j].Y; // Intersection with right edge
        double top_int = (zeta - (C_m*data[j].Y)) / data[i].Y; // Intersection with top edge
        double bottom_int = zeta / data[i].Y; // Intersection with bottom edge

        if(top_int > C/m) {
            // H is the intersection with the right edge
            H = right_int;
        } else if(top_int < 0) {
            // H is the intersection with the left edge
            H = left_int;
        } else {
            // H is at the top edge
            H = C_m;
        }

        if(bottom_int > C_m) {
            // L is the intersection with the right edge
            L = right_int;
        } else if(bottom_int < 0) {
            // L is the intersection with the left edge
            L = left_int;
        } else {
            // L is at the bottom edge
            L = 0;
        }

        return {L, H};
    }

    void optimize(vector<double>& alpha, int i, int j) {
        double zeta = 0;
        for(int u = 0; u < m; u++) {
            if(u == i || u == j) continue;
            zeta -= alpha[u] * data[u].Y;
        }
        // zeta = -\sum_{u=1,u \ne i,j}^{m}\alpha_u y^{(u)}
        // \alpha_i y(i) + \alpha_j y(j) = \zeta

        auto [L, H] = intersect(i, j, zeta); // Restrictions on \alpha_j

        double dA = (2*dotprod(i, j)) - dotprod(i, i) - dotprod(j, j);
        double dB = data[j].Y - data[i].Y + (zeta * dotprod(i, i)) - (zeta * dotprod(i, j));
        for(int k = 0; k < m; k++) {
            if(k == i || k == j) continue;
            dB += alpha[k] * data[k].Y * (dotprod(i, k) - dotprod(j, k));
        }
        dB *= data[j].Y;

        double new_alpha_j;
        if(dA == 0) {
            // A line instead of a parabola.
            // Either alpha_j = H or alpha_j = L is the optimal choice.
            vector<double> new_alpha = alpha;
            new_alpha[j] = H;
            new_alpha[i] = (zeta - new_alpha[j]*data[j].Y)*data[i].Y;
            double evalH = eval(new_alpha);
            new_alpha[j] = L;
            new_alpha[i] = (zeta - new_alpha[j]*data[j].Y)*data[i].Y;
            double evalL = eval(new_alpha);

            new_alpha_j = (evalH > evalL) ? H : L;
        } else {
            new_alpha_j = -(dB / dA);
            if(new_alpha_j > H) new_alpha_j = H;
            if(new_alpha_j < L) new_alpha_j = L;
        }
        
        double new_alpha_i = (zeta - new_alpha_j*data[j].Y) * data[i].Y;

        alpha[j] = new_alpha_j;
        alpha[i] = new_alpha_i;
    }

public:
    SVM(const vector<DataPoint>& _data, int _epochs, double _C) : data(_data), epochs(_epochs), C(_C) {
        m = data.size(); // training data size
        param = data[0].X.size(); // length of vector X
        w = vector<double>(param); // weights
    }

    void calculate() {
        RandomElementGenerator gen(0, m-1); // rng for picking pairs;

        // Finding the optimal alpha coefficients
        vector<double> alpha(m, 0);
        long long i, j;
        for(int x = 0; x < epochs; x++) {
            i = gen.get();
            do { j = gen.get(); } while(i == j);
            // Optimize with respect to the a_i, a_j pair.
            optimize(alpha, i, j);
            if((x+1) % (epochs/20) == 0) {
                cout << "Completed " << x+1 << "/" << epochs << ".\n";
            }
        }

        // Finding the optimal direction w
        for(int u = 0; u < m; u++) {
            for(int v = 0; v < param; v++) {
                w[v] += alpha[u]*data[u].Y*data[u].X[v];
            }
        }
        cout << "Weights found.\n";

        // Finding the optimal bias b
        double C_m = C/m;
        found = false;
        for(int u = 0; u < m && !found; u++) {
            if(alpha[u] > 0 && alpha[u] < C_m) {
                cout << "Bias found.\n";
                b = data[u].Y - dotprod(w, data[u].X);
                found = true;
            }
        }

        if(!found) {
            cout << "Failed to find a satisfactory bias.\n";
        }
    }

    void importWeights(vector<double>& w, double b) {
        if(w.size() != data.size())
        this->w = w; this->b = b;
        found = true;
    }

    optional<pair<vector<double>, double>> exportWeights() {
        if(!found) return nullopt;
        return make_pair(w, b);
    }

    double classify(vector<double>& p) {
        return dotprod(w, p) + b;
    }

    double evaluate(DataPoint& p) {
        return p.Y * (dotprod(w, p.X) + b);
    }
};

#endif