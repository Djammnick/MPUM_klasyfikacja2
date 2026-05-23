#ifndef _RANDOM_HPP_
#define _RANDOM_HPP_

#include <bits/stdc++.h>
using namespace std;

class RandomElementGenerator {
    random_device rd;
    uniform_int_distribution<long long> dist;
    mt19937 mt;
public:
    // Constructs a random number generator from range [lo, hi].
    RandomElementGenerator(long long lo, long long hi) {
        mt = mt19937(rd());
        dist = uniform_int_distribution<long long>(lo, hi);
    }

    // Generates a random number.
    long long get() {
        return dist(mt);
    }
};

#endif