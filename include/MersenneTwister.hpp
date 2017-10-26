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
using clock = chrono::high_resolution_clock;

class MersenneTwister {
 private:
  mt19937_64 myMersenne;
  uniform_real_distribution<double> doubleDist;
  uniform_int_distribution<int> intDist;
 public:
  MersenneTwister() {
    auto seed = clock::now().time_since_epoch().count();
    myMersenne = mt19937_64(seed);
    doubleDist = uniform_real_distribution<double>(0, 1);
    intDist = uniform_int_distribution<int>(0, 1);
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

  vector<int> randomValues(int minValue, int maxValue, int numValues, bool replacement = true) {
    vector<int> myvector(maxValue - minValue);

    if (replacement)
      for (int i = 0; i < myvector.size(); i++)
        myvector[i] = i_random(minValue, maxValue);
    else {
      std::iota(myvector.begin(), myvector.end(), 3);
      shuffle(myvector.begin(), myvector.end(), myMersenne);
    }
  }
};

#endif //MACHINE_LEARNING_MERSENNETWISTER_HPP
