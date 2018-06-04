#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <array>
#include "include/KNN.hpp"
#include "include/matrix/Matrix.hpp"
#include "include/LeastSquares.hpp"
#include "include/PCA.hpp"
#include "include/LDA.hpp"
#include "include/KMeans.hpp"
#include "include/MLP.hpp"
#include "include/ClassifierUtils.hpp"
#include "include/NaiveBayes.hpp"
#include "include/GridWorld.hpp"

using namespace std;
using myClock = chrono::high_resolution_clock;

string datasetDir = "/home/dodo/Documents/FEI/Matérias/PEL 208 - Tópicos Especiais em Aprendizagem/trabalhos/datasets/";

std::vector<double> getNextLineAndSplitIntoTokens(std::istream &str) {
  std::vector<double> result;
  std::string line;
  std::getline(str, line);

  std::stringstream lineStream(line);
  std::string cell;

  while (std::getline(lineStream, cell, ',')) {
    double value = stod(cell);
    result.push_back(value);
  }

  return result;
}

vector<double> csvToRowVector(string path) {
  vector<double> row, aux;

  ifstream arquivo(path);
  if (!arquivo.good())
    throw runtime_error("File doesn't exist");

  while (!(aux = getNextLineAndSplitIntoTokens(arquivo)).empty()) {
    row.push_back(aux[0]);
  }
  return row;
}

vector<vector<double>> csvToVector(string path,
                                   bool normalize = false,
                                   int ignoreColumn = -1) {
  vector<vector<double>> outer;
  vector<double> innerVector;
  vector<double> sums;
  vector<double> means;
  vector<double> dev;

  ifstream arquivo(path);
  if (!arquivo.good())
    throw runtime_error("File '" + path + "' doesn't exist");

  while (!(innerVector = getNextLineAndSplitIntoTokens(arquivo)).empty()) {
    outer.push_back(innerVector);
    if (normalize) {
      if (sums.empty()) {
        for (double x :innerVector)
          sums.push_back(x);
      } else {
        #pragma omp parallel for
        for (int i = 0; i < sums.size(); i++) {
          sums[i] += innerVector[i];
        }
      }
    }
  }

  if (normalize) {
    for (int i = 0; i < sums.size(); i++) {
      means.push_back(sums[i] / outer.size());
      double squaredSum = 0;

      for (auto datum : outer) {
        squaredSum += pow(datum[i] - means[i], 2);
      }

      dev.push_back(sqrt(squaredSum / (outer.size() - 1)));
    }

    #pragma omp parallel for
    for (int j = 0; j < outer.size(); j++) {
      for (int i = 0; i < outer[j].size(); i++) {
        if (i != ignoreColumn) {
          outer[j][i] = (outer[j][i] - means[i]) / dev[i];
        }
      }
    }
  }
  return outer;
}

void testBooks() {
  const vector<vector<double>>
      &data = csvToVector(datasetDir + "books/normalized-training.csv", false);
  const vector<vector<double>>
      &test = csvToVector(datasetDir + "books/normalized-test.csv", false);
  KNN knn(data, 2);

  int ks[] = {1, 2, 3, 5, 10};

  for (auto k:ks) {
    knn.setK(k);
    const vector<double> yPred = knn.regression(test);

    for (const double y : yPred) {
      cout << "\t" << y;
    }

    cout << endl;
  }
}

double accuracy(vector<double> yTrue, vector<double> yPred) {
  int right = 0;

  #pragma omp parallel for reduction(+:right)
  for (int i = 0; i < yTrue.size(); i++)
    right += (yTrue[i] == yPred[i]);

  return right / (double) yTrue.size();
}

void testIris() {
  // 0 = setosa
  // 1 = versicolor
  // 2 = virginica
  const vector<vector<double>>
      &data = csvToVector(datasetDir + "iris/normalized-training.csv", true, 4);
  const vector<vector<double>> &test = csvToVector(datasetDir + "iris/normalized-testing.csv", true);
  const vector<double> &yTrue = csvToRowVector(datasetDir + "iris/normalized-testing-y.csv");

  KNN knn(data, 4);
  int ks[] = {1, 2, 3, 5, 10};
  for (auto k:ks) {
    knn.setK(k);
    const vector<double> yPred = knn.classify(test);
    cout << k << "\t" << accuracy(yTrue, yPred) << endl;
    for (const double y : yPred) {
      cout << "  " << y;
    }
    cout << endl;
  }
}

void testPoker() {
  const vector<vector<double>> &data = csvToVector(datasetDir + "poker-hand/training.csv", false);
  const vector<vector<double>> &test = csvToVector(datasetDir + "poker-hand/testing-single.csv", false);
  const vector<double> &yTrue = csvToRowVector(datasetDir + "poker-hand/testing-single-y.csv");

  KNN knn(data, 10, 1, KNN::Distance::HAMMING);

  int ks[] = {1, 2, 3, 5, 10};

  for (int k:ks) {
    knn.setK(k);
    const vector<double> yPred = knn.classify(test, test.size() >= 1000);
    cout << k << "\t" << accuracy(yTrue, yPred) << endl;
  }
}

void testWine(const string &path) {
  const vector<vector<double>> &data = csvToVector(path + "normalized-training.csv", false);
  const vector<vector<double>> &test = csvToVector(path + "normalized-testing.csv", false);
  const vector<double> &yTrue = csvToRowVector(path + "normalized-testing-y.csv");

  KNN knn(data, 11);

  int ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50};

  for (int k:ks) {
    knn.setK(k);
    const vector<double> yPred = knn.classify(test, test.size() >= 1000);
    cout << k << "\t" << accuracy(yTrue, yPred) << endl;
  }
}

void testAddRowColumn() {
  const double arr[] = {1, 2,
                        3, 4,
                        5, 6};
  MatrixD m(3, 2, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));

  cout << m;
  const double zerosColumn[] = {0, 0, 0};
  MatrixD column(3, 1, vector<double>(zerosColumn, zerosColumn + sizeof(zerosColumn) / sizeof(zerosColumn[0])));
  m.addColumn(column, 1);
  cout << m;
//    m.addRow(0, vector<double>(zerosColumn, zerosColumn + sizeof(zerosColumn) / sizeof(zerosColumn[0])));
//    cout << m;
}

void testInverseDeterminant() {
  const double arr1[] = {3, 5, 2,
                         8, 4, 8,
                         2, 4, 7};
  MatrixD test_d1(3, 3, vector<double>(arr1, arr1 + sizeof(arr1) / sizeof(arr1[0])));
  const double arr2[] = {9, 5, 2, 5,
                         9, 5, 3, 7,
                         6, 5, 4, 8,
                         1, 5, 3, 7};
  MatrixD test_d2(4, 4, vector<double>(arr2, arr2 + sizeof(arr2) / sizeof(arr2[0])));
  const double arr3[] = {3, 6, 2,
                         8, 6, 5,
                         9, 1, 6};
  MatrixD test_d3(3, 3, vector<double>(arr3, arr3 + sizeof(arr3) / sizeof(arr3[0])));

  double d1 = test_d1.determinant(), d2 = test_d2.determinant(), d3 = test_d3.determinant();
  cout << d1 << endl;
  cout << d2 << endl;
  cout << d3 << endl;
  cout << test_d1.inverse();
  cout << test_d2.inverse();
  cout << test_d3.inverse();
}

void testOperations() {
  const double arr[] = {1, 2,
                        3, 4,
                        5, 6};
  MatrixD m(3, 2, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0]))),
      t = m.transpose(), mt = m * t, tm = t * m, mm = m + m, tt = t + t;
  cout << m;
  cout << t;
  cout << mt;
  cout << tm;
  cout << mm;
  cout << tt;
}

void testBigOperations() {
  // good for testing OpenMP implementation
  MersenneTwister twister;
  vector<size_t> sizes = {10, 50, 100, 250, 500, 1000};

  for (auto s: sizes) {
    MatrixD a(s, s, twister.vecFromNormal(s * s));
    MatrixD b(s, s, twister.vecFromNormal(s * s));

    chrono::time_point<chrono::system_clock> start = myClock::now();
    a * b;
    chrono::duration<float> execution_time = myClock::now() - start;
    cout << s << '\t' << execution_time.count() << endl;
  }
}

void testMatrixFromCSV() {
  MatrixD m = MatrixD::fromCSV(datasetDir + "alpswater/alpswater.csv");
  cout << m;
}

void testEigen() {
  const double arr[] = {4, 2, 0,
                        2, 5, 3,
                        0, 3, 6};
  MatrixD coitada(3, 3, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));

  pair<MatrixD, MatrixD> eigens = coitada.eigen();
  MatrixD val = eigens.first, vec = eigens.second;

  cout << val << endl << vec;
}

void testMatrices() {
//    testOperations();
//    testInverseDeterminant();
//    testAddRowColumn();
//    testMatrixFromCSV();
  testEigen();
}

void testLeastSquaresAlps() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "alpswater/alpswater.csv");
  MatrixD X = data.getColumn(0);
  MatrixD y = data.getColumn(1);
  LeastSquares l(X, y);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();

  X.addColumn(X.getColumn(0).hadamard(X.getColumn(0)), 1);
  l = LeastSquares(X, y);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();

  X.removeColumn(1);
  l = LeastSquares(X, y, LeastSquares::WEIGHTED);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();
}

void testLeastSquaresBooks() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "books/training.csv");
  MatrixD y = data.getColumn(2);
  MatrixD X = data;
  X.removeColumn(2);
  LeastSquares l(X, y);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();

  X.addColumn(X.getColumn(0).hadamard(X.getColumn(0)), 2);
  X.addColumn(X.getColumn(1).hadamard(X.getColumn(1)), 3);
  l = LeastSquares(X, y);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();

  X.removeColumn(2);
  X.removeColumn(2);
  l = LeastSquares(X, y, LeastSquares::WEIGHTED);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();
}

void testLeastSquaresCensus() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "us-census/training.csv");
  MatrixD X = data.getColumn(0);
  MatrixD y = data.getColumn(1);
  LeastSquares l(X, y);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();

  X.addColumn(X.getColumn(0).hadamard(X.getColumn(0)), 1);
  l = LeastSquares(X, y);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();

  X.removeColumn(1);
  l = LeastSquares(X, y, LeastSquares::WEIGHTED);
  l.fit();
  cout << "Coefficients" << endl << l.getCoefs() << "Residuals" << endl << l.getResiduals();
}

void testLeastSquares() {
  testLeastSquaresAlps();
  testLeastSquaresBooks();
  testLeastSquaresCensus();
}

void testPCALindsay() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "lindsay.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCAAlps() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "alpswater/alpswater.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCABooks() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "books/training.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCACensus() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "us-census/training.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCAHald() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "hald/hald.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCA() {
  testPCALindsay();
  testPCAAlps();
  testPCABooks();
  testPCACensus();
  testPCAHald();
}

void testLDAIris() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "iris/original.csv");

  MatrixD y = data.getColumn(4);
  data.removeColumn(4);

  LDA lda(data, y);
  lda.fit();

  cout << lda.transform();
}

void testPCAIris() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "iris/original.csv");

  MatrixD y = data.getColumn(4);
  data.removeColumn(4);

  PCA pca(data);
  pca.fit();

  cout << pca.transform(2);
}

void testMDFIris() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "iris/original.csv");

  MatrixD y = data.getColumn(4);
  data.removeColumn(4);

  PCA pca(data);
  pca.fit();
  cout << pca.getCumPercentages();

  LDA lda(pca.transform(), y);
  lda.fit();

  cout << lda.transform();
}

void testLDA() {
//  testLDAIris();
//  testPCAIris();
  testMDFIris();
}

void testKMeansToyDataset() {
  MatrixD
      data = MatrixD::fromCSV(datasetDir + "synth-clustering/kmeans-toy.csv");

  KMeans kmeans;
  kmeans.fit(data, 3, 100, 1, 2, KMeans::RANDOM);
  cout << kmeans.getY();
}

void testKMeansIris() {
  MatrixD data = MatrixD::fromCSV(datasetDir + "iris/original.csv");
  data.removeColumn(4);
  KMeans kmeans;
  kmeans.fit(data, 3);
  cout << kmeans.getY();
}

void testGiantToyDatasets() {
  KMeans kmeans;
  ofstream myfile;

  MatrixD sset = MatrixD::fromCSV(datasetDir + "synth-clustering/s-set.csv");
  kmeans.fit(sset, 15, 100, 100, 2, KMeans::SAMPLE, true);
  myfile.open(datasetDir + "synth-clustering/sset-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();

  MatrixD birch1 = MatrixD::fromCSV(datasetDir + "synth-clustering/birch1.csv");
  kmeans.fit(birch1, 100, 1, 100, 2, KMeans::SAMPLE, true);
  myfile.open(datasetDir + "synth-clustering/birch1-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();

  MatrixD birch2 = MatrixD::fromCSV(datasetDir + "synth-clustering/birch2.csv");
  kmeans.fit(birch2, 100, 1, 100, 2, KMeans::SAMPLE, true);
  myfile.open(datasetDir + "synth-clustering/birch2-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();

  MatrixD birch3 = MatrixD::fromCSV(datasetDir + "synth-clustering/birch3.csv");
  kmeans.fit(birch3, 100, 1, 100, 2, KMeans::SAMPLE, true);
  myfile.open(datasetDir + "synth-clustering/birch3-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();
}

void testKMeans() {
//  testKMeansToyDataset();
//  testKMeansIris();
  testGiantToyDatasets();
}

void sanityCheck() {
  const double arr[] = {9, 1, 1, 2,
                        9, 2, 3, 4,
                        9, 3, 5, 2,
                        9, 4, 7, 4};
  const double y[] = {0, 1, 0, 1};
  MatrixD m1(4, 4, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));
  MatrixD groups(4, 1, vector<double>(y, y + sizeof(y) / sizeof(y[0])));

  cout << m1.mean();
  cout << m1.mean(groups);
}

void testMLPXor() {
  MLP mlp;

  MatrixD dataMatrix(4, 2, {0, 0,
                            0, 1,
                            1, 0,
                            1, 1});

  vector<MatrixD> hidden(2);

  MatrixD hidden1(3, 2, {.25, .25,
                         .05, .10,
                         .15, .2});

  MatrixD hidden2(3, 2, {.45, .45,
                         .25, .3,
                         .35, .4});

  hidden[0] = hidden1;
  hidden[1] = hidden2;

  MatrixD yMatrix(4, 1, {0, 1, 1, 0});

  MatrixD yPred;

  unsigned iters = 1000000, batch = 0;
  double learningRate = 1, minError = .0000001, lambda = 0;
  bool standardize = false, adaptive = true;

  mlp.fit(dataMatrix,
          yMatrix,
          hidden,
          iters,
          batch,
          learningRate,
          minError,
          lambda,
          MLP::SIGMOID,
          adaptive,
          standardize);
  yPred = mlp.predict(dataMatrix, MLP::SUMMARY);
  cout << yPred << "confusion matrix: \n" << ClassifierUtils::confusionMatrix(yMatrix, yPred) << endl
       << "accuracy: " << ClassifierUtils::accuracy(yMatrix, yPred) << endl
       << "precision: " << ClassifierUtils::precision(yMatrix, yPred) << endl
       << "recall: " << ClassifierUtils::recall(yMatrix, yPred) << endl
       << "f_score: " << ClassifierUtils::f_score(yMatrix, yPred) << endl;
}

void testMLPIris() {
  MatrixD trainData = MatrixD::fromCSV(datasetDir + "iris/original.csv");
  MatrixD yTrain = trainData.getColumn(4);
  trainData.removeColumn(4);

  MatrixD testData = MatrixD::fromCSV(datasetDir + "iris/testing.csv");
  MatrixD yTrue = testData.getColumn(4);
  testData.removeColumn(4);

  MatrixD yPred;
  MLP mlp;

  unsigned iters = 40000, batch = 37;
  double learningRate = 1, minError = 0, lambda = 0;
  bool standardize = false, adaptive = false;
  MLP::WeightInitialization wInit = MLP::GLOROT;

  mlp.fit(trainData,
          yTrain,
          vector<size_t>({3}),
          iters,
          batch,
          learningRate,
          minError,
          lambda,
          MLP::SIGMOID,
          wInit,
          adaptive,
          standardize);
  yPred = mlp.predict(testData, MLP::SUMMARY);
  cout << yPred << "confusion matrix: \n" << ClassifierUtils::confusionMatrix(yTrue, yPred) << endl
       << "accuracy: " << ClassifierUtils::accuracy(yTrue, yPred) << endl;

  mlp.fit(trainData,
          yTrain,
          vector<size_t>({4}),
          iters,
          batch,
          learningRate,
          minError,
          lambda,
          MLP::SIGMOID,
          wInit,
          adaptive,
          standardize);
  yPred = mlp.predict(testData, MLP::SUMMARY);
  cout << yPred << "confusion matrix: \n" << ClassifierUtils::confusionMatrix(yTrue, yPred) << endl
       << "accuracy: " << ClassifierUtils::accuracy(yTrue, yPred) << endl;

  mlp.fit(trainData,
          yTrain,
          vector<size_t>({3, 3}),
          iters,
          batch,
          learningRate,
          minError,
          lambda,
          MLP::SIGMOID,
          wInit,
          adaptive,
          standardize);
  yPred = mlp.predict(testData, MLP::SUMMARY);
  cout << yPred << "confusion matrix: \n" << ClassifierUtils::confusionMatrix(yTrue, yPred) << endl
       << "accuracy: " << ClassifierUtils::accuracy(yTrue, yPred) << endl;

  mlp.fit(trainData,
          yTrain,
          vector<size_t>({4, 4}),
          iters,
          batch,
          learningRate,
          minError,
          lambda,
          MLP::SIGMOID,
          wInit,
          adaptive,
          standardize);
  yPred = mlp.predict(testData, MLP::SUMMARY);
  cout << yPred << "confusion matrix: \n" << ClassifierUtils::confusionMatrix(yTrue, yPred) << endl
       << "accuracy: " << ClassifierUtils::accuracy(yTrue, yPred) << endl;
}

void testMLPDigits() {
//  string dataPath = datasetDir + "digits/oneseights/";
  string dataPath = datasetDir + "digits/";

  MatrixD data = MatrixD::fromCSV(dataPath + "train.csv");
  MatrixD y = MatrixD::fromCSV(dataPath + "train_labels.csv");
  MatrixD testData = MatrixD::fromCSV(dataPath + "test.csv");
  MatrixD yTest = MatrixD::fromCSV(dataPath + "test_labels.csv");
  MatrixD yPred;

  MLP mlp;

  unsigned iters = 1000000, batch = 1;
  double learningRate = 1, minError = 0, lambda = .05;
  bool standardize = false, adaptive = true;
  MLP::WeightInitialization wInit = MLP::UNIFORM;

  mlp.fit(data,
          y,
          vector<size_t>({256, 256}),
          iters,
          batch,
          learningRate,
          minError,
          lambda,
          MLP::SIGMOID,
          wInit,
          adaptive,
          standardize);
  yPred = mlp.predict(testData, MLP::SUMMARY);
  cout << "confusion matrix: \n" << ClassifierUtils::confusionMatrix(yTest, yPred) << endl
       << "accuracy: " << ClassifierUtils::accuracy(yTest, yPred) << endl;
}

void testMLP() {
//  testMLPXor();
  testMLPDigits();
//  testMLPIris();
}

void testNaiveBayes() {
  MatrixD data;

  NaiveBayes nb = NaiveBayes(datasetDir + "naivebayes/tennis.csv");
  nb = NaiveBayes(datasetDir + "naivebayes/laptop_phone.csv");
  nb.predict(CSVReader::csvToStringVecVec(datasetDir + "naivebayes/laptop_phone_test.csv"));
  nb = NaiveBayes(datasetDir + "naivebayes/mau_pagador.csv");
  nb.predict(CSVReader::csvToStringVecVec(datasetDir + "naivebayes/mau_pagador_test.csv"));
}

void testDynamicProgramming() {
  vector<pair<size_t, size_t>> goals(2);
  goals[0].first = goals[0].second = 0;
  goals[1].first = goals[1].second = 5;
  size_t gridSize = 6;
  GridWorld d;
  Timer timer;
  double gamma = 1, theta = .000001, alpha = .3, epsilon = .8;
  unsigned int maxIters = 100000;

  for (int i = 0; i < 32; ++i) {
//    timer.start();
//    d.policyIteration(gridSize, gridSize, goals, gamma, theta, i == 0);
//    cout << "policy:\t" << timer.runningTime() << endl;
//    timer.start();
//    d.valueIteration(gridSize, gridSize, goals, gamma, theta, i == 0);
//    cout << "value:\t" << timer.runningTime() << endl;
    timer.start();
    d.MonteCarloEstimatingStarts(gridSize, gridSize, goals, gamma, maxIters, i == 0);
    cout << "mc:\t" << timer.runningTime() << endl;
    timer.start();
    d.Sarsa(gridSize, gridSize, goals, gamma, alpha, epsilon, maxIters, i == 0);
    cout << "sarsa:\t" << timer.runningTime() << endl;
    timer.start();
    d.QLearning(gridSize, gridSize, goals, gamma, alpha, epsilon, maxIters, i == 0);
    cout << "ql:\t" << timer.runningTime() << endl;
  }
}

int main() {
  cout.precision(12);
//  testBooks();
//  testIris();
//  testPoker();
//  string red_path = datasetDir + "winequality-red/";
//  string white_path = datasetDir + "winequality-white/";
//  testWine(red_path);
//  testWine(white_path);
//  testMatrices();
//  testLeastSquares();
//  testPCA();
//  testLDA();
//  testKMeans();
//  testMLP();
//  testNaiveBayes();
  testDynamicProgramming();
//  testBigOperations();
//  sanityCheck();
  return 0;
}

