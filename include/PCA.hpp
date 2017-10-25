//
// Created by dodo on 16/10/17.
//

#ifndef MACHINE_LEARNING_PCA_HPP
#define MACHINE_LEARNING_PCA_HPP

#include "Matrix.hpp"

using namespace std;

class PCA {

 private:
  Matrix X, eigenvalues, eigenvectors, percentages, cumPercentages;
 public :
  explicit PCA(Matrix data) {
    X = std::move(data);
  }

  void fit() {
    Matrix XMinusMean = X.minusMean(); // standardize columns to have 0 mean
    Matrix covariances = XMinusMean.cov(); // get covariance matrix of the data

    // get the sum of variances, this'll be useful later
    double sumVar = 0;
    for (size_t i = 0; i < covariances.nRows(); i++) {
      sumVar += covariances(i, i);
    }

    pair<Matrix, Matrix> eig = covariances.eigen(); // eigenvalues and eigenvectors of cov matrix
    eigenvalues = eig.first;
    eigenvectors = eig.second;

    // calculate the percentage of variance that each eigenvalue "explains"
    percentages = Matrix(eigenvalues.nRows(), eigenvalues.nCols());
    cumPercentages = Matrix(eigenvalues.nRows(), eigenvalues.nCols());
    for (int i = 0; i < eigenvalues.nRows(); i++) {
      percentages(i, 0) = eigenvalues(i, 0) / sumVar;
      cumPercentages(i, 0) = i == 0 ? percentages(i, 0) : percentages(i, 0) + cumPercentages(i - 1, 0);
    }
  }

  Matrix transform() {
    Matrix finalData = eigenvectors.transpose() * X.minusMean().transpose();
    return finalData.transpose();
  }

  Matrix transform(int numComponents) {
    Matrix filter = Matrix::zeros(eigenvalues.nRows(), 1);

    for (int i = 0; i < numComponents; i++) {
      filter(i, 0) = 1;
    }

    Matrix finalData = eigenvectors.getColumns(filter).transpose() * X.minusMean().transpose();
    return finalData.transpose();
  }

  const Matrix &getEigenvalues() const {
    return eigenvalues;
  }

  const Matrix &getEigenvectors() const {
    return eigenvectors;
  }

  const Matrix &getPercentages() const {
    return percentages;
  }

  const Matrix &getCumPercentages() const {
    return cumPercentages;
  }
};

#endif //MACHINE_LEARNING_PCA_HPP
