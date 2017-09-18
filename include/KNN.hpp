//
// Created by dodo on 13/09/17.
//

#ifndef MACHINE_LEARNING_KNN_HPP
#define MACHINE_LEARNING_KNN_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

using namespace std;

class KNN {
 private:
  vector<vector<double>> data;
  int yColumn, k;

  //! Finds the k-nearest neighbors of a data element
  //! \param chosen_indices integer array that will keep the indices of the k-nearest neighbors
  //! \param testie a vector of real values representing a data element
  void getKNN(int *chosen_indices, const vector<double> &testie) {

    // initialize two arrays with indices of the k nearest neighbors in our data
    // and their distances from our new element
    double chosen_distances[k];

    int largestDistanceIndex = 0;
    double largestCandidateDistance = distance(testie, data[0]);
    for (int i = 0; i < data.size(); i ++) {
      // get the euclidean distance from the new element to the current element in our dataset
      double current_distance = distance(testie, data[i]);

      // first, fill the array with the first k candidates
      if (i < k) {
        chosen_indices[i] = i;
        chosen_distances[i] = current_distance;

        if (current_distance >= largestCandidateDistance) {
          largestDistanceIndex = i;
          largestCandidateDistance = current_distance;
        }
      }
        // then, look for the furthest neighbor candidate and check if ours is closer
      else if (current_distance < largestCandidateDistance) {

        //if it is, replace the furthest one with the current one
        chosen_indices[largestDistanceIndex] = i;
        chosen_distances[largestDistanceIndex] = current_distance;

        // look, in the k candidates, the furthest one
        largestDistanceIndex = 0;
        largestCandidateDistance = chosen_distances[0];
        for (int j = 1; j < k; j ++) {
          if (largestCandidateDistance < chosen_distances[j]) {
            largestDistanceIndex = j;
            largestCandidateDistance = chosen_distances[j];
          }
        }
      }
    }
  }

 public:
  int getK() const {
    return k;
  }

  void setK(int k) {
    KNN::k = k;
  }

  const vector<vector<double>> &getData() const {
    return data;
  }

  int getYColumn() const {
    return yColumn;
  }

  explicit KNN(vector<vector<double>> data, int yColumn, int k = 1) {
    this->data = std::move(data);

    sort(this->data.begin(), this->data.end(), [](const vector<double> &a, const vector<double> &b) {
      for (int i = 0; i < a.size(); i ++) {
        if (a[i] < b[i])
          return true;
        if (a[i] > b[i])
          return false;
      }
      return false;
    });

    this->yColumn = yColumn;
    this->k = k;
  }

  //! Calculates the Euclidean distance between two vectors
  //! \param a first vector
  //! \param b second vector
  //! \return Euclidean distance between a and b
  double distance(vector<double> a, vector<double> b) {
    double d = 0;
    for (int i = 0; i < a.size(); i ++) {
      // ignore our dependent variable column
      if (i == yColumn)
        continue;

      d += pow(a[i] - b[i], 2);
    }
    return sqrt(d);
  }

  double regression(const vector<double> &testie) {
    // get the indices of the k-nearest neighbors
    int chosen_indices[k];
    getKNN(chosen_indices, testie);

    // sum the y column
    double ySum = 0;

#pragma omp parallel for reduction(+:ySum)
    for (int j = 0; j < k; j ++) {
      ySum += data[chosen_indices[j]][yColumn];
    }

    // return its mean value
    return ySum / k;
  }

  double classify(const vector<double> &testie) {
    // get the indices of the k-nearest neighbors
    int chosen_indices[k];
    getKNN(chosen_indices, testie);

    // initialize vectors that will hold individual classes and their number of appearances in the knn
    int class_votes[k] = {0};
    vector<double> classes;

    // sum the occurrences of each class
    class_votes[0] = 1;
    classes.push_back(data[chosen_indices[0]][yColumn]);
    for (int j = 1; j < k; j ++) {
      double current_class = data[chosen_indices[j]][yColumn];
      for (int i = 0; i < classes.size(); i ++) {
        if (classes[i] == current_class) {
          class_votes[i] ++;
          break;
        }
      }
    }

    // get the class with the most votes
    double winner = classes[0];
    int winner_votes = class_votes[0];
    for (int j = 1; j < k; j ++) {
      if (class_votes[j] > winner_votes) {
        winner = classes[j];
        winner_votes = class_votes[j];
      }

    }

    return winner;
  }
};

#endif
