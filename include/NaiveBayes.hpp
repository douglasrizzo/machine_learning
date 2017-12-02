/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief Naive Bayes classifier
 * @date   2017-11-17
 */

#ifndef MACHINE_LEARNING_NAIVEBAYES_HPP
#define MACHINE_LEARNING_NAIVEBAYES_HPP

using namespace std;

#include <string>
#include <map>
#include "Matrix.hpp"
#include "strasser/csv.h"

using namespace std;

class NaiveBayes {
 private:
  MatrixI lookupTable;
  vector<string> lookupColumns, lookupRows;
 public:
  NaiveBayes(string csvPath, vector<string> xs, string y) {
    lookupColumns = vector<string>(), lookupRows = vector<string>();

    vector<string> csvHeader = xs;
    csvHeader.push_back(y);

    vector<string> csvRow;

    io::CSVReader<csvHeader.size()> in(csvPath);
    in.read_header(io::ignore_extra_column, csvHeader);
    while (in.read_row(csvRow)) {
      for (int i = 0; i < csvRow.size() - 1; i++)
        lookupColumns.push_back(csvHeader[i] + '+'csvRow[i]);

      lookupRows.push_back(csvHeader[csvHeader.size() - 1] + '+'
                               + csvRow[csvHeader.size() - 1]);
    }

    sort(lookupRows.begin(), lookupRows.end());
    sort(lookupColumns.begin(), lookupColumns.end());
    lookupRows.erase(unique(lookupRows.begin(), lookupRows.end()), lookupRows.end());
    lookupColumns.erase(unique(lookupColumns.begin(), lookupColumns.end()), lookupColumns.end());

    lookupTable = MatrixI::zeros(lookupRows.size(),
                                 lookupColumns.size());

    in = io::CSVReader<csvHeader.size()>(csvPath);
    in.read_header(io::ignore_extra_column, csvHeader);
    while (in.read_row(csvRow)) {
      for (int i = 0; i < csvRow.size() - 1; i++) {
        size_t row = static_cast<size_t>(distance(lookupRows.begin(), find(lookupRows.begin(), lookupRows.end(),
                                                                           csvHeader[i] + '+'csvRow[i])));
        size_t
            col = static_cast<size_t>(distance(lookupColumns.begin(), find(lookupColumns.begin(), lookupColumns.end(),
                                                                           csvHeader[csvHeader.size() - 1] + '+'
                                                                               + csvRow[csvHeader.size() - 1])));

        lookupTable(row, col) += 1;
      }
    }
  }

  vector<string> predict(map<string, vector<string>> X) {
  }
};

#endif //MACHINE_LEARNING_NAIVEBAYES_HPP
