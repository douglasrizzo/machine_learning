//
// Created by dodo on 25/10/17.
//

#ifndef MACHINE_LEARNING_KMEANS_HPP
#define MACHINE_LEARNING_KMEANS_HPP

#include "Matrix.hpp"
#include "MersenneTwister.hpp"

class KMeans {
 public:
  enum InitializationMethod { RANDOM, SAMPLE };
 private:
  MatrixD X, y, centroids;
  unsigned int k, totalIterations;
  double distance, sse;
  int verbosity;
  InitializationMethod initMethod;

  static MatrixD minkowski(MatrixD m, double p, bool root = true) {
    MatrixD distances = MatrixD::zeros(m.nRows(), m.nRows());

    for (size_t i = 0; i < m.nRows(); i++) {
      for (size_t j = i + 1; j < m.nRows(); j++) {
        for (size_t k = 0; k < m.nCols(); k++) {
          distances(i, j) += pow(abs(m(i, k) - m(j, k)), p);
        }
        if (root)
          distances(i, j) = pow(distances(i, j), 1 / p);
      }
    }

    return distances;
  }

  static MatrixD minkowski(MatrixD a, MatrixD b, double p, bool root = true) {
    if (a.nCols() != b.nCols())
      throw runtime_error("Matrices have different number of dimensions");

    MatrixD distances = MatrixD::zeros(a.nRows(), b.nRows());

    for (size_t i = 0; i < a.nRows(); i++) {
      for (size_t j = 0; j < b.nRows(); j++) {
        for (size_t k = 0; k < a.nCols(); k++) {
          double x1 = a(i, k), x2 = b(j, k);
          double term = pow(abs(x1 - x2), p);
          distances(i, j) += term;
        }

        if (root)
          distances(i, j) = pow(distances(i, j), 1 / p);
      }
    }

    return distances;
  }

  static MatrixD chebyshev(MatrixD a, MatrixD b) {
    if (a.nCols() != b.nCols())
      throw runtime_error("Matrices have different number of dimensions");

    MatrixD distances = MatrixD::zeros(a.nRows(), b.nRows());

    for (size_t i = 0; i < a.nRows(); i++) {
      for (size_t j = 0; j < b.nRows(); j++) {
        for (size_t k = 0; k < a.nCols(); k++) {
          double x1 = a(i, k), x2 = b(j, k);
          double term = abs(x1 - x2);
          distances(i, j) = max(distances(i, j), term);
        }
      }
    }

    return distances;
  }

  static MatrixD chebyshev(MatrixD a) {
    MatrixD distances = MatrixD::zeros(a.nRows(), a.nRows());

    for (size_t i = 0; i < a.nRows(); i++) {
      for (size_t j = i + 1; j < a.nRows(); j++) {
        for (size_t k = 0; k < a.nCols(); k++) {
          double x1 = a(i, k), x2 = a(j, k);
          double term = abs(x1 - x2);
          distances(i, j) = distances(j, i) = max(distances(i, j), term);
        }
      }
    }

    return distances;
  }

  static MatrixD euclidean(MatrixD m, bool root = true) {
    return minkowski(m, 2, root);
  }

  static MatrixD manhattan(MatrixD m, bool root = true) {
    return minkowski(m, 1, root);
  }

  static MatrixD euclidean(MatrixD a, MatrixD b, bool root = true) {
    return minkowski(a, b, 2, root);
  }

  static MatrixD manhattan(MatrixD a, MatrixD b, bool root = true) {
    return minkowski(a, b, 1, root);
  }

  static MatrixD normalizeToSmallestInt(MatrixD m) {
    if (!m.isColumn())
      throw runtime_error("Nor a column vector");

    MatrixD result(m.nRows(), 1);
    MatrixD u = m.unique();
    u.sort();

    return result;
  }

  double SSE() {
    return euclidean(X, centroids, false).sum();
  }

 public:
  KMeans() {}

  MatrixD predict(MatrixD data) {
    MatrixD distances = minkowski(X, centroids, distance, true);
//    Matrix distances = euclidean(X, centroids, false);
    MatrixD results = MatrixD::zeros(X.nRows(), 1);

    for (size_t i = 0; i < distances.nRows(); i++) {
      for (size_t j = 1; j < distances.nCols(); j++) {
        double current_c = results(i, 0);
        double current_d = distances(i, current_c);
        double new_c = j;
        double new_d = distances(i, new_c);
        if (new_d < current_d)
          results(i, 0) = new_c;
      }
    }

    return results;
  }

  void fit(MatrixD data,
           unsigned int k,
           unsigned int iters = 100,
           unsigned int inits = 100,
           double distance = 2,
           InitializationMethod initMethod = SAMPLE, bool verbose = false) {
    this->X = data.standardize();
    this->k = k;
    this->initMethod = initMethod;
    this->distance = distance;
    this->totalIterations = 0;
    MersenneTwister twister;
    double minSSE;

    for (int currentInit = 0; currentInit < inits; currentInit++) {

      if (initMethod == RANDOM) {
        centroids = MatrixD(k, X.nCols());
        for (size_t i = 0; i < centroids.nRows(); i++)
          for (size_t j = 0; j < centroids.nCols(); j++)
            centroids(i, j) = twister.d_random(X.getColumn(j).min(),
                                               X.getColumn(j).max());
      } else {
        vector<int> sample = twister.randomValues(X.nRows(), k, false);
        centroids = MatrixD();
        for (int i = 0; i < sample.size(); i++)
          centroids.addRow(X.getRow(sample[i]));
      }

      MatrixD yPrev;

      for (int currentIteration = 0; currentIteration < iters; currentIteration++) {
        if (verbose)
          cout << currentInit << '/' << inits << ' ' << currentIteration << '/' << iters << endl;

        MatrixD yCurr = predict(X); // assignment

        if (yCurr == yPrev)
          break;

        yPrev = yCurr;
        centroids = X.mean(yCurr); // centroid update
      }
      if (currentInit == 0 or SSE() < minSSE) {
        minSSE = SSE();
        y = yPrev;
      }
    }
    this->sse = minSSE;
  }

  const MatrixD &getY() const {
    return y;
  }

  const MatrixD &getCentroids() const {
    return centroids;
  }

  unsigned int getK() const {
    return k;
  }

  unsigned int getTotalIterations() const {
    return totalIterations;
  }

  double getDistance() const {
    return distance;
  }

  double getSse() const {
    return sse;
  }

};

#endif //MACHINE_LEARNING_KMEANS_HPP
