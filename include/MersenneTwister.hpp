/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief
 * @date   2017-10-25
 */

#ifndef MACHINE_LEARNING_MERSENNETWISTER_HPP
#define MACHINE_LEARNING_MERSENNETWISTER_HPP

#include <random>
#include <chrono>
#include <algorithm>

using namespace std;

class MersenneTwister {
  using clock = chrono::high_resolution_clock;
 private:
  mt19937_64 myMersenne;
  uniform_real_distribution<double> doubleDist;
  uniform_int_distribution<int> intDist;
  normal_distribution<double> normalDist;
 public:
  MersenneTwister() {
    auto seed = clock::now().time_since_epoch().count();
    myMersenne = mt19937_64(seed);
    doubleDist = uniform_real_distribution<double>(0, 1);
    intDist = uniform_int_distribution<int>(0, 1);
    normalDist = normal_distribution<double>(0, 1);
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \return double between 0 and 1
  double d_random() {
    return doubleDist(myMersenne);
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \param min the lower bound for the random number
//! \param max the upper bound for the random number
//! \return double between min and max
  double d_random(double min, double max) {
    return d_random() * (max - min) + min;
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \param max the upper bound for the random number
//! \return double between 0 and max
  double d_random(double max) {
    return d_random() * max;
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \return double between 0 and 1
  int i_random() {
    return intDist(myMersenne);
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \param min the lower bound for the random number
//! \param max the upper bound for the random number
//! \return double between min and max
  int i_random(int min, int max) {
    return i_random() * (max - min) + min;
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \param max the upper bound for the random number
//! \return double between 0 and max
  double i_random(int max) {
    return i_random() * max;
  }

  vector<int> randomValues(int maxValue, unsigned int numValues, bool replacement = true) {
    return randomValues(0, maxValue, numValues, replacement);
  }

  vector<int> randomValues(int minValue, int maxValue, unsigned int numValues, bool replacement = true) {
    vector<int> myvector(maxValue - minValue);
    if (replacement)
      for (int &i : myvector)
        i = i_random(minValue, maxValue);
    else {
      iota(myvector.begin(), myvector.end(), minValue);
      shuffle(myvector.begin(), myvector.end(), myMersenne);
      myvector = vector<int>(&myvector[0], &myvector[numValues]);
    }

    return myvector;
  }

  double n_random() {
    return normalDist(myMersenne);
  }

  vector<double> vecFromNormal(int n) {
    vector<double> myvector(n);
    for (double &i : myvector)
      i = n_random();

    return myvector;
  }
};

#endif //MACHINE_LEARNING_MERSENNETWISTER_HPP
