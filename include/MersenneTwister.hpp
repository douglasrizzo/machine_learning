/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Pseudo-random number generator using the Mersenne twister
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
  uniform_real_distribution<double> uniformDoubleDist;
  uniform_int_distribution<int> uniformIntDist;
  normal_distribution<double> normalDist;
 public:
  MersenneTwister() {
    auto seed = clock::now().time_since_epoch().count();
    myMersenne = mt19937_64(seed);
    uniformDoubleDist = uniform_real_distribution<double>(0, 1);
    uniformIntDist = uniform_int_distribution<int>(0, 1);
    normalDist = normal_distribution<double>(0, 1);
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \return double between 0 and 1
  double d_random() {
    uniformDoubleDist = uniform_real_distribution<double>(0, 1);
    return uniformDoubleDist(myMersenne);
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
//! \return integer between 0 and 1
  int i_random() {
    uniformIntDist = uniform_int_distribution<int>(0, 1);
    return uniformIntDist(myMersenne);
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \param min the lower bound for the random number
//! \param max the upper bound for the random number
//! \return integer between min and max
  int i_random(int min, int max) {
    uniformIntDist = uniform_int_distribution<int>(min, max);
    return uniformIntDist(myMersenne);
  }

//! Pseudo-random number generator using the Mersenne Twister method
//! \param max the upper bound for the random number
//! \return integer between 0 and max
  int i_random(int max) {
    return i_random(0, max);
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
    normalDist = normal_distribution<double>(0, 1);
    return normalDist(myMersenne);
  }

  double n_random(double mean, double stddev) {
    normalDist = normal_distribution<double>(mean, stddev);
    return normalDist(myMersenne);
  }

  vector<double> vecFromNormal(size_t n) {
    vector<double> myvector(n);
    for (double &i : myvector)
      i = n_random();

    return myvector;
  }

  vector<double> vecFromNormal(size_t n, double mean, double stddev) {
    normalDist = normal_distribution<double>(mean, stddev);
    vector<double> myvector(n);
    for (double &i : myvector)
      i = normalDist(myMersenne);

    return myvector;
  }

  vector<double> vecFromUniform(size_t n) {
    vector<double> myvector(n);
    for (double &i : myvector)
      i = d_random();

    return myvector;
  }

  vector<double> vecFromUniform(size_t n, double min, double max) {
    uniformDoubleDist = uniform_real_distribution<double>(min, max);
    vector<double> myvector(n);
    for (double &i : myvector)
      i = uniformDoubleDist(myMersenne);

    return myvector;
  }
  unsigned int uint_random(unsigned int max) {
    uniform_int_distribution<unsigned int> uniformUIntDist;
    uniformUIntDist = uniform_int_distribution<unsigned int>(0, max);
    return uniformUIntDist(myMersenne);
  }
};

#endif //MACHINE_LEARNING_MERSENNETWISTER_HPP
