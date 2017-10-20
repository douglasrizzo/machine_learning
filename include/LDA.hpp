//
// Created by dodo on 18/10/17.
//

#ifndef MACHINE_LEARNING_LDA_HPP
#define MACHINE_LEARNING_LDA_HPP

#include <utility>

#include "Matrix.hpp"

using namespace std;

class LDA {
 private:
  Matrix X, y, withinSpread, betweenSpread;
 public:
  LDA(Matrix data, Matrix classes) : X(std::move(data)), y(std::move(classes)) {

  }

  void fit() {
    Matrix innerMean = X.mean(y); // means for each class
    Matrix grandMean = X.mean(); // mean of the entire data set

    // calculate covariance matrices of each class
    vector<Matrix> covariances;
    for (int i = 0; i < y.unique().nRows(); i++) {
      Matrix currentCov = X.getRows(y == i).cov();
      covariances.push_back(currentCov);
    }

    // calculate within-class spread

    // calculate between-class spread

    // fit the new line
  }

  Matrix predict(Matrix data) {
    return Matrix::fill(data.nRows(), 1, -1);
  }
};

#endif //MACHINE_LEARNING_LDA_HPP
