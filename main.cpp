#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <array>
#include "include/KNN.hpp"
#include "include/Matrix.hpp"
#include "include/LeastSquares.hpp"
#include "include/PCA.hpp"
#include "include/LDA.hpp"
#include "include/KMeans.hpp"

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

    return outer;
  }
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
  for (int i = 0; i < yTrue.size(); i++)
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

void testAddRowColumn() {
  const double arr[] = {1, 2,
                        3, 4,
                        5, 6};
  Matrix<double> m(3, 2, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));

  cout << m;
  const double zerosColumn[] = {0, 0, 0};
  Matrix<double> column(3, 1, vector<double>(zerosColumn, zerosColumn + sizeof(zerosColumn) / sizeof(zerosColumn[0])));
  m.addColumn(column, 1);
  cout << m;
//    m.addRow(0, vector<double>(zerosColumn, zerosColumn + sizeof(zerosColumn) / sizeof(zerosColumn[0])));
//    cout << m;
}

void testInverseDeterminant() {
  const double arr1[] = {3, 5, 2,
                         8, 4, 8,
                         2, 4, 7};
  Matrix<double> test_d1(3, 3, vector<double>(arr1, arr1 + sizeof(arr1) / sizeof(arr1[0])));
  const double arr2[] = {9, 5, 2, 5,
                         9, 5, 3, 7,
                         6, 5, 4, 8,
                         1, 5, 3, 7};
  Matrix<double> test_d2(4, 4, vector<double>(arr2, arr2 + sizeof(arr2) / sizeof(arr2[0])));
  const double arr3[] = {3, 6, 2,
                         8, 6, 5,
                         9, 1, 6};
  Matrix<double> test_d3(3, 3, vector<double>(arr3, arr3 + sizeof(arr3) / sizeof(arr3[0])));

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
  Matrix<double> m(3, 2, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0]))),
      t = m.transpose(), mt = m * t, tm = t * m, mm = m + m, tt = t + t;
  cout << m;
  cout << t;
  cout << mt;
  cout << tm;
  cout << mm;
  cout << tt;
}

void testMatrixFromCSV() {
  Matrix<double> m = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/alpswater/alpswater.csv");
  cout << m;
}

void testEigen() {
  const double arr[] = {4, 2, 0,
                        2, 5, 3,
                        0, 3, 6};
  Matrix<double> coitada(3, 3, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));

  pair<Matrix<double>, Matrix<double>> eigens = coitada.eigen();
  Matrix<double> val = eigens.first, vec = eigens.second;

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
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/alpswater/alpswater.csv");
  Matrix<double> X = data.getColumn(0);
  Matrix<double> y = data.getColumn(1);
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
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/books/training.csv");
  Matrix<double> y = data.getColumn(2);
  Matrix<double> X = data;
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
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/us-census/training.csv");
  Matrix<double> X = data.getColumn(0);
  Matrix<double> y = data.getColumn(1);
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
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/lindsay.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCAAlps() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/alpswater/alpswater.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCABooks() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/books/training.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCACensus() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/us-census/training.csv");
  PCA pca(data);
  pca.fit();
//    cout << pca.getEigenvalues().transpose() << pca.getPercentages().transpose() * 100
//         << pca.getCumPercentages().transpose() * 100 << endl;
  cout << pca.getEigenvectors() << endl << pca.transform() << endl;
}

void testPCAHald() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/hald/hald.csv");
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
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/iris/original.csv");

  Matrix<double> y = data.getColumn(4);
  data.removeColumn(4);

  LDA lda(data, y);
  lda.fit();

  cout << lda.transform();
}

void testPCAIris() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/iris/original.csv");

  Matrix <double> y = data.getColumn(4);
  data.removeColumn(4);

  PCA pca(data);
  pca.fit();

  cout << pca.transform(2);
}

void testMDFIris() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/iris/original.csv");

  Matrix<double> y = data.getColumn(4);
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
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/synth-clustering/kmeans-toy.csv");

  KMeans kmeans;
  kmeans.fit(data, 3, 100, 1, 2, KMeans::RANDOM);
  cout << kmeans.getY();
}

void testKMeansIris() {
  Matrix<double> data = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/iris/original.csv");
  data.removeColumn(4);
  KMeans kmeans;
  kmeans.fit(data, 3);
  cout << kmeans.getY();
}

void testGiantToyDatasets() {
  KMeans kmeans;
  ofstream myfile;

  Matrix<double> sset = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/synth-clustering/s-set.csv");
  kmeans.fit(sset, 15, 100, 100, 2, KMeans::SAMPLE, true);
  myfile.open("/home/dodo/repos/machine_learning/datasets/synth-clustering/sset-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();

  Matrix<double> birch1 = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/synth-clustering/birch1.csv");
  kmeans.fit(birch1, 100, 1, 100, 2, KMeans::SAMPLE, true);
  myfile.open("/home/dodo/repos/machine_learning/datasets/synth-clustering/birch1-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();

  Matrix<double> birch2 = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/synth-clustering/birch2.csv");
  kmeans.fit(birch2, 100, 1, 100, 2, KMeans::SAMPLE, true);
  myfile.open("/home/dodo/repos/machine_learning/datasets/synth-clustering/birch2-clusters.txt");
  myfile << kmeans.getY();
  myfile.close();

  Matrix<double> birch3 = Matrix<double>::fromCSV("/home/dodo/repos/machine_learning/datasets/synth-clustering/birch3.csv");
  kmeans.fit(birch3, 100, 1, 100, 2, KMeans::SAMPLE, true);
  myfile.open("/home/dodo/repos/machine_learning/datasets/synth-clustering/birch3-clusters.txt");
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
  Matrix<double> m1(4, 4, vector<double>(arr, arr + sizeof(arr) / sizeof(arr[0])));
  Matrix<double> groups(4, 1, vector<double>(y, y + sizeof(y) / sizeof(y[0])));

  cout << m1.mean();
  cout << m1.mean(groups);
}

int main() {
  cout.precision(12);
//  testBooks();
//  testIris();
//  testPoker();
//  string red_path = "/home/dodo/repos/machine_learning/datasets/winequality-red/";
//  string white_path = "/home/dodo/repos/machine_learning/datasets/winequality-white/";
//  testWine(red_path);
//  testWine(white_path);
//  testMatrices();
//  testLeastSquares();
//  testPCA();
//  testLDA();
  testKMeans();
//  sanityCheck();
  return 0;
}

