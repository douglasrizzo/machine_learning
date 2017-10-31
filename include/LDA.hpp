/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Linear discriminant analysis algorithm
 * @date   2017-10-18
 */

#ifndef MACHINE_LEARNING_LDA_HPP
#define MACHINE_LEARNING_LDA_HPP

#include <utility>

#include "Matrix.hpp"

using namespace std;

/**
 * Linear discriminant analysis algorithm
 */
class LDA {
 private:
  MatrixD X, y, eigenvalues, eigenvectors;
 public:
  /**
   * Linear discriminant analysis algorithm
   * @param data The matrix whose linear discriminants will be found
   * @param classes Column vector containing the classes each row element in <code>data</code> belongs to
   */
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
