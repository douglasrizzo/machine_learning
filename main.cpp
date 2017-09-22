#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <array>
#include "include/KNN.hpp"

using namespace std;

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
  if (! arquivo.good())
    throw runtime_error("File doesn't exist");

  while (! (aux = getNextLineAndSplitIntoTokens(arquivo)).empty()) {
    row.push_back(aux[0]);
  }
  return row;
}

vector<vector<double>> csvToVector(string path,
                                   bool normalize = false,
                                   int ignoreColumn = - 1) {
  vector<vector<double>> outer;
  vector<double> innerVector;
  vector<double> sums;
  vector<double> means;
  vector<double> dev;

  ifstream arquivo(path);
  if (! arquivo.good())
    throw runtime_error("File '" + path + "' doesn't exist");

  while (! (innerVector = getNextLineAndSplitIntoTokens(arquivo)).empty()) {
    outer.push_back(innerVector);
    if (normalize) {
      if (sums.empty()) {
        for (double x :innerVector)
          sums.push_back(x);
      }

      else {
#pragma omp parallel for
        for (int i = 0; i < sums.size(); i ++) {
          sums[i] += innerVector[i];
        }
      }
    }
  }

  if (normalize) {
    for (int i = 0; i < sums.size(); i ++) {
      means.push_back(sums[i] / outer.size());
      double squaredSum = 0;

      for (auto datum : outer) {
        squaredSum += pow(datum[i] - means[i], 2);
      }

      dev.push_back(sqrt(squaredSum / (outer.size() - 1)));
    }

#pragma omp parallel for
    for (int j = 0; j < outer.size(); j ++) {
      for (int i = 0; i < outer[j].size(); i ++) {
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
      &data = csvToVector("/home/dodo/repos/machine_learning/datasets/books/normalized-training.csv", false);
  const vector<vector<double>>
      &test = csvToVector("/home/dodo/repos/machine_learning/datasets/books/normalized-test.csv", false);
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
  for (int i = 0; i < yTrue.size(); i ++)
    right += (yTrue[i] == yPred[i]);

  return right / (double) yTrue.size();
}

void testIris() {
  // 0 = setosa
  // 1 = versicolor
  // 2 = virginica
  const vector<vector<double>>
      &data = csvToVector("/home/dodo/repos/machine_learning/datasets/iris/normalized-training.csv", true, 4);
  const vector<vector<double>> &test =
      csvToVector("/home/dodo/repos/machine_learning/datasets/iris/normalized-testing.csv", true);
  const vector<double>
      &yTrue = csvToRowVector("/home/dodo/repos/machine_learning/datasets/iris/normalized-testing-y.csv");

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
  const vector<vector<double>>
      &data = csvToVector("/home/dodo/repos/machine_learning/datasets/poker-hand/training.csv", false);
  const vector<vector<double>>
      &test = csvToVector("/home/dodo/repos/machine_learning/datasets/poker-hand/testing-single.csv", false);
  const vector<double>
      &yTrue = csvToRowVector("/home/dodo/repos/machine_learning/datasets/poker-hand/testing-single-y.csv");

  KNN knn(data, 10, 1, KNN::Distance::HAMMING);

  int ks[] = {1, 2, 3, 5, 10};

  for (int k:ks) {
    knn.setK(k);
    const vector<double> yPred = knn.classify(test, test.size() >= 1000);
    cout << k << "\t" << accuracy(yTrue, yPred) << endl;
  }
}

void testWine(const string &path) {
  const vector<vector<double>>
      &data = csvToVector(path + "normalized-training.csv", false);
  const vector<vector<double>>
      &test = csvToVector(path + "normalized-testing.csv", false);
  const vector<double>
      &yTrue = csvToRowVector(path + "normalized-testing-y.csv");

  KNN knn(data, 11);

  int ks[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50};

  for (int k:ks) {
    knn.setK(k);
    const vector<double> yPred = knn.classify(test, test.size() >= 1000);
    cout << k << "\t" << accuracy(yTrue, yPred) << endl;
  }
}

int main() {
//  cout.precision(12);
  testBooks();
//  testIris();
//  testPoker();
//  string red_path = "/home/dodo/repos/machine_learning/datasets/winequality-red/";
//  string white_path = "/home/dodo/repos/machine_learning/datasets/winequality-white/";
//  testWine(red_path);
//  testWine(white_path);

  return 0;
}