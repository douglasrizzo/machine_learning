//
// Created by dodo on 25/10/17.
//

#ifndef MACHINE_LEARNING_KMEANS_HPP
#define MACHINE_LEARNING_KMEANS_HPP

#include "Matrix.hpp"

class KMeans {
 public:
  enum InitializationMethod { RANDOM, SAMPLE };
 private:
  Matrix X, y, centroids;
  unsigned int k, iters;
  InitializationMethod initMethod;
 public:
  KMeans() {}

  Matrix fit(Matrix X, unsigned int k, InitializationMethod initMethod = RANDOM) {
    this->X = X;
    this->k = k;
    this->initMethod = initMethod;

    switch (initMethod) {
      case RANDOM:
//        centroids=
        break;
      case SAMPLE:

        break;
    }

    y = predict(X);
    Matrix cur_y;

    do {
      y = cur_y;
      centroids = X.mean(y);
      cur_y = predict(X);
    } while (y != cur_y);
  }
};

#endif //MACHINE_LEARNING_KMEANS_HPP
