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
  Matrix X, y, eigenvalues, eigenvectors;
 public:
  LDA(Matrix data, Matrix classes) : X(std::move(data)), y(std::move(classes)) {

  }

  void fit() {
    MatrixD Sw = X.WithinClassScatter(y);
    MatrixD Sb = X.BetweenClassScatter(y);

    auto eigen = (Sw.inverse() * Sb).eigen();

    eigenvalues = eigen.first;
    eigenvectors = eigen.second;
  }

  Matrix transform() {
    Matrix finalData = eigenvectors.transpose() * X.transpose();
    return finalData.transpose();
  }

  Matrix predict(Matrix data) {
    return Matrix::fill(data.nRows(), 1, -1);
  }
};

#endif //MACHINE_LEARNING_LDA_HPP
