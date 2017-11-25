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
  MatrixD data, classes, dataMean, dataDev;
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
           double errorThreshold = 0.0001,
           ActivationFunction func = SIGMOID,
           bool standardize = true,
           bool verbose = true) {
    size_t outputEncodingSize = y.unique().nRows();

    // number of layers. even if there are no hidden layers,
    // there will exist at least one layer of weights that will need to be fitted
    size_t nLayers = hiddenConfig.size() < 1 ? 1 : hiddenConfig.size() + 1;
    // initialize vector of weight matrices
    vector<MatrixD> w = vector<MatrixD>(nLayers);

    // initialize weights
    for (int i = 0; i < nLayers; i++) {
      size_t nIn, nOut;

      // number of inputs (+1 accounts for the weight used in the bias)
      nIn = (i == 0 ? X : w[i - 1]).nCols() + 1;

      // number of outputs
      nOut = i == w.size() - 1 ? outputEncodingSize : hiddenConfig[i];

      w[i] = initUniform(nIn, nOut); // initialize layer with random numbers from a distribution
    }

    fit(X, y, w, maxIters, learningRate, errorThreshold, func, standardize);
  }

  void fit(MatrixD X,
           MatrixD y,
           vector<MatrixD> hiddenLayers,
           int maxIters,
           double learningRate = 0.01,
           double errorThreshold = 0.0001,
           ActivationFunction func = SIGMOID,
           bool standardize = true,
           bool verbose = true) {
    // create one-hot encoding for classes
    classes = y.oneHot();
    size_t outputEncodingSize = classes.nCols();

    for (int i = 0; i < hiddenLayers.size(); ++i) {
      size_t correct_nIn = (i == 0 ? X : hiddenLayers[i - 1]).nCols() + 1,
          correct_nOut = i == hiddenLayers.size() - 1 ? outputEncodingSize : hiddenLayers[i + 1].nRows() - 1;
      if (hiddenLayers[i].nRows() != correct_nIn) {
        throw invalid_argument(
            "Weight matrix " + to_string(i) + " input (" + to_string(hiddenLayers[i].nRows()) + ") should be ("
                + to_string(correct_nIn) + ")");
      }
      if (hiddenLayers[i].nCols() != correct_nOut) {
        throw invalid_argument(
            "Weight matrix " + to_string(i) + " output (" + to_string(hiddenLayers[i].nCols()) + ") should be ("
                + to_string(correct_nOut) + ")");
      }
    }

    W = hiddenLayers;

    // number of layers. even if there are no hidden layers,
    // there will exist at least one layer of weights that will need to be fitted
    size_t nLayers = hiddenLayers.size();

    if (standardize) {
      // if standardization takes place, mean and stddev are stored to be used in future predictions
      dataMean = X.mean();
      dataDev = X.stdev();
      data = X.standardize(dataMean, dataDev);
    } else
      data = X;

    function<double(double)> activationFunction, activationDerivative;
    if (func == SIGMOID) {
      activationFunction = sigmoid;
      activationDerivative = sigmoidDerivative;
    } else {
      activationFunction = tanh;
      activationDerivative = tanhDerivative;
    }

    double previousLoss;

    // training iterations
    for (int iter = 0; iter < maxIters; iter++) {
      // matrices used in forward pass
      // Z holds f(S), where f is the activation function
      vector<MatrixD> Z(nLayers);
      // F are activation derivatives
      vector<MatrixD> F(nLayers);

      // matrices used on backpropagation
      // D contains the loss signals for each layer
      vector<MatrixD> D(nLayers);

      MatrixD currentInput = data;
      //forward pass
      for (int i = 0; i < nLayers; i++) {
        // add the bias column to the input of the current layer
        currentInput.addColumn(MatrixD::ones(currentInput.nRows(), 1), 0);

        MatrixD S = currentInput * W[i]; // multiply input by weights
        F[i] = S.apply(activationDerivative).transpose();

        //apply activation function, whose resulting matrix will be the next input
        currentInput = Z[i] = S.apply(activationFunction);
      }

      // backpropagation
      // last layer error signal
      D[nLayers - 1] = (Z[nLayers - 1] - classes).transpose();

      // calculate loss
      double loss = 0;
      for (size_t i = 0; i < data.nRows(); i++) {
        double thisError = 0;
        for (size_t j = 0; j < outputEncodingSize; j++) {
          loss += pow(classes(i, j) - Z[nLayers - 1](i, j), 2);
        }
      }
      loss /= (2 * data.nRows());

      if (iter == 0)
        previousLoss = loss;

      if (verbose) {
        char errorChar = loss == previousLoss ? '=' : loss > previousLoss ? '+' : '-';
        cout << "it " << iter + 1 << ", loss: " << loss << ' ' << errorChar << endl;
      }
      if (loss < errorThreshold) {
        break;
      }
      previousLoss = loss;

      // in case there is nothing wrong with our loss, continue backpropagation
      // loss signals for the intermediate layers
      for (int i = nLayers - 2; i >= 0; i--) {
        MatrixD W_noBias = W[i + 1].transpose();
        W_noBias.removeColumn(0);
        W_noBias = W_noBias.transpose();
        // mxb  mxb            mxn       nxb
        D[i] = F[i].hadamard(W_noBias * D[i + 1]);
      }

      // weight updates
      for (size_t i = 0; i < nLayers; i++) {
        MatrixD input = i == 0 ? data : Z[i - 1];
        input.addColumn(MatrixD::ones(input.nRows(), 1), 0);
        //      bxm                       mxb         bxm
        //               scalar       mxn     nxb
        MatrixD dW = -learningRate * (D[i] * input).transpose();
        // bxm
        W[i] += dW;
      }
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