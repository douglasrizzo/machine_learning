/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Multilayer perceptron
 * @date   2017-11-1
 */

#ifndef MACHINE_LEARNING_MLP_HPP
#define MACHINE_LEARNING_MLP_HPP

#include <vector>
#include <chrono>
#include <iostream>
#include "../include/matrix/Matrix.hpp"
#include "../include/mersenne_twister/MersenneTwister.hpp"
#include "Timer.hpp"

using namespace std;
using myClock = chrono::high_resolution_clock;

/**
 * Multi-layer perceptron
 */
class MLP {
 private:
  MatrixD data, dataMean, dataDev, classes, originalClasses;
  vector<MatrixD> W;

  //region Activation functions

  //! @param x
  //! @return x to the power of 2
  static double pow2(double x) {
    return x * x;
  }

  //! @param x
  //! @return Sigmoid of x
  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  //! @param x
  //! @return Derivative of the sigmoid of x
  static double sigmoidDerivative(double x) {
    double z = sigmoid(x);
    return z * (1 - z);
  }

  //! @param x
  //! @return Hyperbolic tangent of x
  static double tanh(double x) {
    return 2 * sigmoid(2 * x) - 1;
  }

  //! @param x
  //! @return Derivative of the hyperbolic tangent of x
  static double tanhDerivative(double x) {
    return 1 - pow(tanh(x), 2);
  }

  //endregion

  //! Initialize a matrix according to a normal distribution N(0; 1)
  //! @param in number of rows
  //! @param out number of columns
  //! @return a matrix of doubles, initialized according to the distribution
  static MatrixD initNormal(size_t in, size_t out) {
    MersenneTwister twister;
    MatrixD result(in, out, twister.vecFromNormal(in * out));
    return result;
  }

  //! Initialize a matrix according to a normal distribution N(mean; stddev)
  //! @param in number of rows
  //! @param out number of columns
  //! @param mean mean of the normal distribution
  //! @param stddev standard deviation of the normal distribution
  //! @return a matrix of doubles, initialized according to the distribution
  static MatrixD initNormal(size_t in, size_t out, double mean, double stddev) {
    MersenneTwister twister;
    MatrixD result(in, out, twister.vecFromNormal(in * out, mean, stddev));
    return result;
  }

  //! Initialize a matrix according to the uniform distribution U(0;1)
  //! @param in number of rows
  //! @param out number of columns
  //! @return a matrix of doubles, initialized according to the distribution
  static MatrixD initUniform(size_t in, size_t out) {
    MersenneTwister twister;
    MatrixD result(in, out, twister.vecFromUniform(in * out));
    return result;
  }

  //! Initialize a matrix according to the uniform distribution U(min; max)
  //! @param in number of rows
  //! @param out number of columns
  //! @param min minimum value
  //! @param max maximum value
  //! @return a matrix of doubles, initialized according to the distribution
  static MatrixD initUniform(size_t in, size_t out, double min, double max) {
    MersenneTwister twister;
    MatrixD result(in, out, twister.vecFromUniform(in * out, min, max));
    return result;
  }

  static MatrixD binarize(MatrixD m) {
    for (size_t i = 0; i < m.nRows(); i++) {
      size_t largest = 0;
      for (size_t j = 0; j < m.nCols(); j++)
        if (m(i, j) > m(i, largest)) largest = j;

      for (size_t j = 0; j < m.nCols(); j++) {
        m(i, j) = j == largest;
      }
    }
    return m;
  }

  MatrixD summarize(MatrixD m) {
    MatrixD result(m.nRows(), 1);

    for (size_t i = 0; i < m.nRows(); i++) {
      size_t largest = 0;
      for (size_t j = 0; j < m.nCols(); j++)
        if (m(i, j) > m(i, largest)) largest = j;

      result(i, 0) = originalClasses(largest, 0);
    }

    return result;
  }

  static MatrixD softmax(MatrixD m) {
    m = m.apply(static_cast<double (*)(double)>(exp));
    for (size_t i = 0; i < m.nRows(); i++) {
      double sum = 0;

      for (size_t j = 0; j < m.nCols(); j++)
        sum += m(i, j);

      for (size_t j = 0; j < m.nCols(); j++)
        m(i, j) /= sum;
    }
    return m;
  }

 public:

  enum ActivationFunction { SIGMOID, TANH };
  enum WeightInitialization { NORMAL, UNIFORM, GLOROT };
  enum OutputFormat { ACTIVATION, SOFTMAX, ONEHOT, SUMMARY };

  MLP() {
  }

  //! Train a multiplayer perceptron
  //! @param X Input data, with rows representing examples and columns representing features
  //! @param y Input labels as a column vector
  //! @param hiddenConfig vector containing the number of neurons in each hidden layer
  //! @param maxIters maximum number of training iterations
  //! @param batchSize size of the batch. If <= 0, the whole data is used in every iteration
  //! @param learningRate learning rate
  //! @param errorThreshold minimum error for early stopping
  //! @param regularization the regularization parameter. 0 indicates no regularization.
  //! @param func activation function
  //! @param weightInit weight initilization procedure
  //! @param adaptiveLR if true, the learning rate linearly decreases according to the number of iterations
  //! @param standardize if true, data is standardized according to its mean and standard deviation
  //! @param verbose output training summary at each iteration
  void fit(MatrixD X,
           MatrixD y,
           vector<size_t> hiddenConfig,
           int maxIters,
           size_t batchSize = 0,
           double learningRate = 0.01,
           double errorThreshold = 0.0001,
           double regularization = 0,
           ActivationFunction func = SIGMOID,
           WeightInitialization weightInit = UNIFORM,
           bool adaptiveLR = false,
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

      // number of inputs (+1 accounts for the bias weight)
      nIn = (i == 0 ? X : w[i - 1]).nCols() + 1;

      // number of outputs
      nOut = i == w.size() - 1 ? outputEncodingSize : hiddenConfig[i];

      // initialize layer with random numbers from a distribution
      if (weightInit == UNIFORM) {
        w[i] = initUniform(nIn, nOut);
      } else if (weightInit == NORMAL) {
        w[i] = initNormal(nIn, nOut);
      } else if (weightInit == GLOROT) {
        w[i] = initUniform(nIn, nOut, -sqrt(nIn), sqrt(nIn));
      }
    }

    fit(X,
        y,
        w,
        maxIters,
        batchSize,
        learningRate,
        errorThreshold,
        regularization,
        func,
        adaptiveLR,
        standardize,
        verbose);
  }

  //! Train a multiplayer perceptron
  //! @param X Input data, with rows representing examples and columns representing features
  //! @param y Input labels as a column vector
  //! @param hiddenLayers a vector of matrices, each one containing the weights for a hidden layer
  //! @param maxIters maximum number of training iterations
  //! @param batchSize size of the batch. If <= 0, the whole data is used in every iteration
  //! @param learningRate learning rate
  //! @param errorThreshold minimum error for early stopping
  //! @param regularization the regularization parameter. 0 indicates no regularization.
  //! @param func activation function
  //! @param adaptiveLR if true, the learning rate linearly decreases according to the number of iterations
  //! @param standardize if true, data is standardized according to its mean and standard deviation
  //! @param verbose output training summary at each iteration
  void fit(MatrixD X,
           MatrixD y,
           vector<MatrixD> hiddenLayers,
           unsigned int maxIters,
           size_t batchSize = 0,
           double learningRate = 0.01,
           double errorThreshold = 0.0001,
           double regularization = 0,
           ActivationFunction func = SIGMOID,
           bool adaptiveLR = false,
           bool standardize = true,
           bool verbose = true) {
    // create one-hot encoding for classes
    classes = y.oneHot();
    originalClasses = y.unique();
    originalClasses.sort();
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
    } else {
      data = X;
      dataMean = dataDev = MatrixD();
    }

    function<double(double)> activationFunction, activationDerivative;
    if (func == SIGMOID) {
      activationFunction = sigmoid;
      activationDerivative = sigmoidDerivative;
    } else {
      activationFunction = tanh;
      activationDerivative = tanhDerivative;
    }

    float lastStdout = 0;
    double previousLoss;
    Timer timer(1, maxIters);
    timer.start();
    // training iterations
    for (int iter = 0; iter < maxIters; iter++) {
      chrono::time_point<chrono::system_clock> iterStart = myClock::now();

      // matrices used in forward pass
      // Z holds the outputs of each layer
      vector<MatrixD> Z(nLayers);

      // F are activation derivatives, they are needed for all but the last layer
      vector<MatrixD> F(nLayers - 1);

      // matrices used on backpropagation
      // D contains the loss signals for each layer
      vector<MatrixD> D(nLayers);

      Matrix<int> filter;

      MatrixD currentInput;
      if (batchSize > 0) {
        MersenneTwister t;
        MatrixI indices(batchSize, 1, t.randomValues(0, data.nRows(), batchSize, false));

        filter = MatrixI::zeros(data.nRows(), 1);

        for (size_t i = 0; i < indices.nRows(); i++) {
          filter(indices(i, 0), 0) = 1;
        }

        currentInput = data.getRows(filter);
      } else
        currentInput = data;

      //forward pass
      for (int i = 0; i < nLayers; i++) {
        // add the bias column to the input of the current layer
        currentInput.addColumn(MatrixD::ones(currentInput.nRows(), 1), 0);

        MatrixD S = currentInput * W[i]; // multiply input by weights

        // calculate derivatives
        if (i < nLayers - 1) // derivative of the last layer is not used, so no need to do it
          F[i] = S.apply(activationDerivative).transpose();

        //apply activation function, whose resulting matrix will be the next input
        currentInput = Z[i] = S.apply(activationFunction);
      }

      // backpropagation
      // last layer error signal
      MatrixD batchClasses = filter.isEmpty() ? classes : classes.getRows(filter);
      D[nLayers - 1] = (Z[nLayers - 1] - batchClasses).transpose();

      // calculate loss
      double loss = (D[nLayers - 1]).apply(pow2).sum() / (2 * batchClasses.nRows());

      double regularizationTerm = 0;
      for (auto w:W)
        regularizationTerm += w.apply(pow2).sum();
      regularizationTerm = regularization > 0 ? regularization / (2 * batchClasses.nRows()) : 0;

      loss += regularizationTerm;

      // error signals for the intermediate layers
      for (int i = nLayers - 2; i >= 0; i--) {
        MatrixD W_noBias = W[i + 1].transpose();
        W_noBias.removeColumn(0);
        W_noBias = W_noBias.transpose();
        // mxb  mxb            mxn       nxb
        D[i] = F[i].hadamard(W_noBias * D[i + 1]);
      }

      // learning rate is linearly scaled down with passing iterations
      double lr = adaptiveLR ? (learningRate / maxIters) * (maxIters - iter) : learningRate;

      // weight updates
      for (size_t i = 0; i < nLayers; i++) {
        MatrixD input;
        if (i == 0)
          if (filter.isEmpty())
            input = data;
          else
            input = data.getRows(filter);
        else
          input = Z[i - 1];

        input.addColumn(MatrixD::ones(input.nRows(), 1), 0); // add the bias once again
        MatrixD dW = -lr * (D[i] * input).transpose();
//        W[i] += dW;
        W[i] = (1 - ((learningRate * regularization) / batchClasses.nRows())) * W[i] + dW;
      }

      if (verbose and timer.activate(iter)) {
        char errorChar = (loss == previousLoss or iter == 0) ? '=' : loss > previousLoss ? '+' : '-';
        cout << "loss: " << loss << ' ' << errorChar << endl;
      }

      if (loss < errorThreshold)
        break;

      previousLoss = loss;
    }
    if (verbose)
      cout << "Total training time: " << timer.runningTime() << endl;
  }

  //! Predict the classes of a data set
  //! @param X Input data to be classified
  //! @param of output format of the method
  //! @return a matrix, each row containing the output of the network for an example of X
  MatrixD predict(MatrixD X, OutputFormat of = ACTIVATION) {
    if (!dataMean.isEmpty() && !dataDev.isEmpty())
      X = X.standardize(dataMean, dataDev);

    // even when there are no hidden layers, there
    // must be at least one of each of the following
    size_t nLayers = W.size();

    MatrixD currentInput = X;
    for (int i = 0; i < nLayers; i++) {
      // add the bias column to the input of the current layer
      currentInput.addColumn(MatrixD::ones(currentInput.nRows(), 1), 0);
      MatrixD S = currentInput * W[i];
      currentInput = S.apply(sigmoid);
    }

    if (of == SOFTMAX)
      return MLP::softmax(currentInput);
    if (of == ONEHOT)
      return MLP::binarize(currentInput);
    if (of == SUMMARY)
      return MLP::summarize(currentInput);

    return currentInput;
  }
};

#endif //MACHINE_LEARNING_MLP_HPP
