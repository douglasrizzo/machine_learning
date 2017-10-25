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
    Matrix innerMean = X.mean(y); // means for each class
    Matrix grandMean = X.mean(); // mean of the entire data set
    Matrix uniqueClasses = y.unique();

    Matrix Sw = Matrix::zeros(X.nCols(), X.nCols()); // within-class scatter matrix
    Matrix Sb = Matrix::zeros(X.nCols(), X.nCols()); // between-class scatter matrix
    for (size_t i = 0; i < uniqueClasses.nRows(); i++) {
      Matrix classElements = X.getRows(y == i); // get class elements

      Matrix scatterMatrix = classElements.scatter();
      Sw += scatterMatrix;

      Matrix meanDiff = innerMean.getRow(i) - grandMean;
      Sb += classElements.nRows() * meanDiff * meanDiff.transpose();
    }

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
