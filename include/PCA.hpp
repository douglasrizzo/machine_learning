//
// Created by dodo on 16/10/17.
//

#ifndef MACHINE_LEARNING_PCA_HPP
#define MACHINE_LEARNING_PCA_HPP

#include "Matrix.hpp"

using namespace std;

class PCA {

 private:
  MatrixD X, eigenvalues, eigenvectors, percentages, cumPercentages;
 public :
  explicit PCA(MatrixD data) {
    X = std::move(data);
  }

  void fit() {
    MatrixD XMinusMean = X.minusMean(); // standardize columns to have 0 mean
    MatrixD covariances = XMinusMean.cov(); // get covariance matrix of the data

    // get the sum of variances, this'll be useful later
    double sumVar = 0;
    for (size_t i = 0; i < covariances.nRows(); i++) {
      sumVar += covariances(i, i);
    }

    pair<MatrixD, MatrixD> eig = covariances.eigen(); // eigenvalues and eigenvectors of cov matrix
    eigenvalues = eig.first;
    eigenvectors = eig.second;

    // calculate the percentage of variance that each eigenvalue "explains"
    percentages = MatrixD(eigenvalues.nRows(), eigenvalues.nCols());
    cumPercentages = MatrixD(eigenvalues.nRows(), eigenvalues.nCols());
    for (int i = 0; i < eigenvalues.nRows(); i++) {
      percentages(i, 0) = eigenvalues(i, 0) / sumVar;
      cumPercentages(i, 0) = i == 0 ? percentages(i, 0) : percentages(i, 0) + cumPercentages(i - 1, 0);
    }
  }

  MatrixD transform() {
    MatrixD finalData = eigenvectors.transpose() * X.minusMean().transpose();
    return finalData.transpose();
  }

  MatrixD transform(int numComponents) {
    MatrixD filter = MatrixD::zeros(eigenvalues.nRows(), 1);

    for (int i = 0; i < numComponents; i++) {
      filter(i, 0) = 1;
    }

    MatrixD finalData = eigenvectors.getColumns(filter).transpose() * X.minusMean().transpose();
    return finalData.transpose();
  }

  const MatrixD &getEigenvalues() const {
    return eigenvalues;
  }

  const MatrixD &getEigenvectors() const {
    return eigenvectors;
  }

  const MatrixD &getPercentages() const {
    return percentages;
  }

  const MatrixD &getCumPercentages() const {
    return cumPercentages;
  }
};

#endif //MACHINE_LEARNING_PCA_HPP
