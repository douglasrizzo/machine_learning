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
  MatrixD X, y, eigenvalues, eigenvectors;
 public:
  LDA(MatrixD data, MatrixD classes) : X(std::move(data)), y(std::move(classes)) {

  }

  void fit() {
    MatrixD Sw = X.WithinClassScatter(y);
    MatrixD Sb = X.BetweenClassScatter(y);

    auto eigen = (Sw.inverse() * Sb).eigen();

    eigenvalues = eigen.first;
    eigenvectors = eigen.second;
  }

  MatrixD transform() {
    MatrixD finalData = eigenvectors.transpose() * X.transpose();
    return finalData.transpose();
  }

  MatrixD predict(MatrixD data) {
    return MatrixD::fill(data.nRows(), 1, -1);
  }
};

#endif //MACHINE_LEARNING_LDA_HPP
