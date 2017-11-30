/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief Naive Bayes classifier
 * @date   2017-11-17
 */

#ifndef MACHINE_LEARNING_NAIVEBAYES_HPP
#define MACHINE_LEARNING_NAIVEBAYES_HPP

using namespace std;

#include <string>
#include <map>
#include "Matrix.hpp"

class NaiveBayes {
  map<string, vector<string>> data;
  vector<string> y;

  NaiveBayes(map<string, vector<string>> X, vector<string> y) : data(X), y(y) {
  }

  vector<string> predict(map<string, vector<string>> X) {
    vector<string> result;

    for(auto const &x : X) {

    }

    return result;
  }
};

#endif //MACHINE_LEARNING_NAIVEBAYES_HPP
