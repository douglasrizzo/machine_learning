//
// Created by dodo on 01/11/17.
//

#ifndef MACHINE_LEARNING_MLP_HPP
#define MACHINE_LEARNING_MLP_HPP

#include <vector>
#include "Matrix.hpp"
#include "MersenneTwister.hpp"

using namespace std;

class MLP {
 private:
  MatrixD data, classes, uniqueClasses, dataMean, dataDev;
  vector<MatrixD> W;
  vector<size_t> hiddenConfig;

  //region Activation functions

  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  static double sigmoidDerivative(double x) {
    double z = sigmoid(x);
    return z * (1 - z);
  }

  static double tanh(double x) {
    return 2 * sigmoid(2 * x) - 1;
  }

  static double tanhDerivative(double x) {
    return 1 - pow(tanh(x), 2);
  }

  //endregion

  static MatrixD initNormal(size_t in, size_t out) {
    MersenneTwister twister;
    MatrixD result(in, out, twister.vecFromNormal(in * out));
    return result;
  }

  static MatrixD initUniform(size_t in, size_t out) {
    MersenneTwister twister;
    MatrixD result(in, out, twister.vecFromUniform(in * out));
    return result;
  }

 public:

  enum ActivationFunction { SIGMOID, TANH };

  MLP() {
  }

  void fit(MatrixD X,
           MatrixD y,
           vector<size_t> hiddenConfig,
           int maxIters,
           double learningRate = 0.01,
           bool standardize = true) {
    // create one-hot encoding for classes
    classes = y.oneHot();
    uniqueClasses = y.unique().oneHot();

    if (standardize) {
      // if standardization takes place, mean and stddev are stored to be used in future predictions
      dataMean = X.mean();
      dataDev = X.stdev();
      data = X.standardize(dataMean, dataDev);
    } else
      data = X;

    // number of layers. even if there are no hidden layers,
    // there will exist at least one layer of weights that will need to be fitted
    size_t nLayers = hiddenConfig.size() < 1 ? 1 : hiddenConfig.size();

    // initialize vector of weight matrices
    W = vector<MatrixD>(nLayers);

    //initialize weights
    for (int i = 0; i < nLayers; i++) {
      // number of inputs (+1 accounts for the weight used in the bias)
      size_t nIn = (i == 0 ? data : W[i - 1]).nCols() + 1,
          nOut = i == W.size() - 1 ? uniqueClasses.nRows() : hiddenConfig[i]; // number of outputs

      W[i] = initNormal(nIn, nOut); // initialize layer with random numbers from normal distribution
    }

    // training iterations
    for (int iter = 0; iter < maxIters; iter++) {
      // Z holds f(S), where f is the activation function
      vector<MatrixD> Z = vector<MatrixD>(nLayers);
      // F are activation derivatives
      vector<MatrixD> F = vector<MatrixD>(nLayers);

      MatrixD currentInput = X;
      //forward pass
      for (int i = 0; i < nLayers; i++) {
        // add the bias column to the input of the current layer
        currentInput.addColumn(MatrixD::ones(currentInput.nRows(), 1), 0);

        MatrixD S = currentInput * W[i]; // multiply input by weights
        F[i] = S.apply(sigmoidDerivative); //

        //apply activation function, whose resulting matrix will be the next input
        currentInput = Z[i] = S.apply(sigmoid);
      }

      // backpropagation

      // D contains the error signals for each layer
      vector<MatrixD> D(nLayers);
      // dW keeps the weight updates
      vector<MatrixD> dW(nLayers);

      // error signals are calculated backwards
      D[nLayers - 1] = (Z[nLayers - 1] - classes).transpose();

      for (int i = nLayers - 2; i >= 0; i--) {
        D[i] = (F[i].hadamard(W[i] * D[i + 1]));
      }

      // weight updates
      dW[0] = -learningRate * (D[0] * data);
      for (size_t i = 1; i < nLayers; i++) {
        dW[i] = -learningRate * (D[i] * Z[i - 1]);
      }

      // TODO update the damn weights!
    }
  }

  MatrixD predict(MatrixD X, bool standardize = false) {
    if (standardize)
      X = X.standardize(dataMean, dataDev);

    // even when there are no hidden layers, there must be at least one of each of the following
    int n = hiddenConfig.size() < 1 ? 1 : hiddenConfig.size();

    MatrixD currentInput = X;
    for (int i = 0; i < n; i++) {
      // add the bias column to the input of the current layer
      currentInput.addColumn(MatrixD::ones(currentInput.nRows(), 1), 0);
      MatrixD S = currentInput * W[i];
      currentInput = S.apply(sigmoid);
    }

    return currentInput;
  }
};

#endif //MACHINE_LEARNING_MLP_HPP