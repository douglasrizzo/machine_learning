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
  // This checks for a trailing comma with no data after it.
//  if (! lineStream && cell.empty()) {
  // If there was a trailing comma then add an empty element.
//    result.push_back(0);
//  }
  return result;
}

vector<vector<double>> csvToVector(string path, bool normalize = false, int ignoreColumn = - 1) {
  vector<vector<double>> outer;
  vector<double> innerVector;
  vector<double> sums;
  vector<double> means;
  vector<double> dev;

  ifstream arquivo(path);
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
  const vector<vector<double>> &data = csvToVector("/home/dodo/Desktop/data/books.csv", true, 2);
  const vector<vector<double>> &test = csvToVector("/home/dodo/Desktop/data/books_test.csv", true, 4);
  KNN knn(data, 2);

  int ks[] = {1, 2, 3, 5, 10};

  for (auto k:ks) {
    knn.setK(k);
    cout << "k = " << k << endl;
    for (const auto testie: test) {
      cout << "  " << knn.regression(testie) << endl;
    }
  }
}

void testIris() {
  // 0 = setosa
  // 1 = versicolor
  // 2 = virginica
  const vector<vector<double>> &data = csvToVector("/home/dodo/Desktop/data/iris.csv");
  const vector<vector<double>> &test = csvToVector("/home/dodo/Desktop/data/iris_test.csv");

  KNN knn(data, 4);
  int ks[] = {1, 2, 3, 5, 10};
  for (auto k:ks) {
    knn.setK(k);
    cout << "k = " << k << endl;
    for (const auto testie:test) {
      cout << "  " << knn.classify(testie) << endl;
    }
  }
}

int main() {
  testBooks();
  testIris();

  return 0;
}