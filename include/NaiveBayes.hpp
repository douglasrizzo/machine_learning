/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief Naive Bayes classifier
 * @date   2017-11-17
 */

#ifndef MACHINE_LEARNING_NAIVEBAYES_HPP
#define MACHINE_LEARNING_NAIVEBAYES_HPP

#include <string>
#include <map>
#include "Matrix.hpp"

using namespace std;

class NaiveBayes {
 private:
  MatrixI lookupTable, yFrequency;
  vector<string> lookupColumns, lookupRows;
 public:

  explicit NaiveBayes(const string &csvPath, bool verbose = true) {
    lookupColumns = vector<string>(), lookupRows = vector<string>();

    vector<vector<string>> data = CSVReader::csvToStringVecVec(csvPath, true);
    vector<string> csvHeader = data[0];
    data.erase(data.begin());

    for (vector<string> csvRow:data) {
      for (int i = 0; i < csvRow.size() - 1; i++)
        lookupRows.push_back(csvHeader[i] + '+' + csvRow[i]);

      lookupColumns.push_back(csvHeader[csvHeader.size() - 1] + '+'
                                  + csvRow[csvHeader.size() - 1]);
    }

    sort(lookupRows.begin(), lookupRows.end());
    sort(lookupColumns.begin(), lookupColumns.end());
    lookupRows.erase(unique(lookupRows.begin(), lookupRows.end()), lookupRows.end());
    lookupColumns.erase(unique(lookupColumns.begin(), lookupColumns.end()), lookupColumns.end());

    lookupTable = MatrixI::zeros(lookupRows.size(),
                                 lookupColumns.size());
    yFrequency = MatrixI::zeros(lookupColumns.size(), 1);

    for (vector<string> csvRow:data) {
      for (int i = 0; i < csvRow.size() - 1; i++) {
        string rowElement = csvHeader[i] + '+' + csvRow[i],
            relevantHeader = csvHeader[csvHeader.size() - 1] + '+' + csvRow[csvHeader.size() - 1];
        size_t row = static_cast<size_t>(distance(lookupRows.begin(),
                                                  find(lookupRows.begin(), lookupRows.end(), rowElement)));
        size_t col = static_cast<size_t>(distance(lookupColumns.begin(),
                                                  find(lookupColumns.begin(), lookupColumns.end(), relevantHeader)));

        lookupTable(row, col) += 1;
        yFrequency(col, 0) += 1;
      }
    }

    if (verbose) {
      cout << "Lookup table:" << endl << lookupTable << endl << "Rows:" << endl;
      for (auto s : lookupRows)
        cout << s << '\t';
      cout << endl << "Columns:" << endl;

      for (auto s : lookupColumns)
        cout << s << '\t';
      cout << endl;

      cout << "Class frequency:" << endl << yFrequency;
    }
  }

  vector<string> predict(vector<vector<string>> data, bool verbose = true) {
    vector<string> csvHeader = data[0];
    data.erase(data.begin());
    vector<string> result(data.size());
    MatrixD probabilities = MatrixD::ones(data.size(), lookupColumns.size());

    // for each line in our test dataset...
    #pragma omp parallel for if(data.size() > 500)
    for (size_t i = 0; i < data.size(); i++) {
      vector<string> csvRow = data[i];

      // for each feature in the current row...
      for (size_t j = 0; j < csvRow.size(); j++) {
        string rowElement = csvHeader[j] + '+' + csvRow[j];
        size_t row = static_cast<size_t>(distance(lookupRows.begin(),
                                                  find(lookupRows.begin(), lookupRows.end(), rowElement)));

        // for each possible outcome...
        for (size_t col = 0; col < lookupColumns.size(); col++) {
          int lookup = lookupTable(row, col), yFreq = yFrequency(col, 0);
          double currentFrequency = (double) lookupTable(row, col) / yFrequency(col, 0);
          probabilities(i, col) *= currentFrequency;
        }
      }

      int maxProbIndex = -1;
      double currentMaxProb = 0, probSum = 0;
      for (size_t j = 0; j < lookupColumns.size(); j++) {
        probSum += probabilities(i, j);
        if (probabilities(i, j) > currentMaxProb) {
          currentMaxProb = probabilities(i, j);
          maxProbIndex = static_cast<int>(j);
        }
      }

      // normalize probabilities so their sum equals 1
      for (size_t j = 0; j < lookupColumns.size(); j++)
        probabilities(i, j) /= probSum;

      result[i] = maxProbIndex != -1 ? lookupColumns[maxProbIndex] : "NaN";
    }

    if (verbose)
      cout << "Probabilities:" << endl << probabilities;

    return result;
  }
};

#endif //MACHINE_LEARNING_NAIVEBAYES_HPP
