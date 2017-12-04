//
// Created by dodo on 03/12/17.
//

#ifndef MACHINE_LEARNING_CSVREADER_HPP
#define MACHINE_LEARNING_CSVREADER_HPP

#include <cstdio>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

class CSVReader {
 public:
  //! Helper function that separates lines in a CSV file into tokens
  //! \param str stream containing the contents of the CSV file
  //! \return a vector with the flattened contents of the CSV converted to double
  static vector<double> csvLineToDoubles(istream &str) {
    vector<string> strVec = csvLineToStrings(str);
    vector<double> result(strVec.size());

    for (int i = 0; i < strVec.size(); i++)
      result[i] = stod(strVec[i]);

    return result;
  }
  //! Helper function that separates lines in a CSV file into tokens
  //! \param str stream containing the contents of the CSV file
  //! \return a vector with the flattened contents of the CSV converted to double
  static vector<string> csvLineToStrings(istream &str, char token = ',') {
    vector<string> result;
    string line;
    getline(str, line);

    stringstream lineStream(line);
    string cell;

    int lineCount = 1;
    while (getline(lineStream, cell, token)) {
      try {
        result.push_back(cell);
      }
      catch (exception ex) {
        cout << ex.what() << " " << cell << " " << lineCount << endl;
        throw ex;
      }
    }

    return result;
  }

  static vector<vector<string>> csvToStringVecVec(string path, bool checkRowLength = false) {
    vector<vector<string>> outer;
    vector<string> innerVector;

    ifstream arquivo(path);
    if (!arquivo.good())
      throw invalid_argument("File '" + path + "' doesn't exist");

    unsigned long numCols = 0;

    while (!(innerVector = CSVReader::csvLineToStrings(arquivo)).empty()) {
      if (numCols == 0)
        numCols = innerVector.size();
      else if (checkRowLength and numCols != innerVector.size())
        throw runtime_error("File has missing values in some columns");
      outer.push_back(innerVector);
    }

    return outer;
  }

  static vector<vector<double>> csvToNumericVecVec(const string &path, bool checkRowLength = false) {
    vector<vector<double>> outer;
    vector<double> innerVector;

    ifstream arquivo(path);
    if (!arquivo.good())
      throw invalid_argument("File '" + path + "' doesn't exist");

    unsigned long numCols = 0;

    while (!(innerVector = CSVReader::csvLineToDoubles(arquivo)).empty()) {
      if (numCols == 0)
        numCols = innerVector.size();
      else if (checkRowLength and numCols != innerVector.size())
        throw runtime_error("File has missing values in some columns");
      outer.push_back(innerVector);
    }

    return outer;
  }
};

#endif //MACHINE_LEARNING_CSVREADER_HPP
