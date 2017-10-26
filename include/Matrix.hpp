/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Matrix implementation, with a series of linear algebra functions
 * @date   2017-10-17
 */

#ifndef MACHINE_LEARNING_MATRIX_HPP
#define MACHINE_LEARNING_MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <set>
#include <complex>
#include "nr3.h"
#include "eigen_unsym.h"

using namespace std;

//! Matrix implementation, with a series of linear algebra functions
class Matrix {
 private:
  size_t mRows;
  size_t mCols;
  std::vector<double> mData;

  //! Validates if indices are contained inside the matrix
  //! \param row row index
  //! \param col column index
  //! \throws runtime error if at least one of the indices is out of bounds
  void validateIndexes(size_t row, size_t col) const {
    if (row < 0 or row >= mRows)
      throw runtime_error(
          "Invalid row index (" + to_string(row) + "): should be between 0 and " + to_string(mRows - 1));
    if (col < 0 or col >= mCols)
      throw runtime_error(
          "Invalid column index (" + to_string(col) + "): should be between 0 and " + to_string(mCols - 1));
  }

  //! Helper function that separates lines in a CSV file into tokens
  //! \param str stream containing the contents of the CSV file
  //! \return a vector with the flattened contents of the CSV converted to double
  static std::vector<double> getNextLineAndSplitIntoTokens(std::istream &str) {
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

  //! Sorts eigenvalues by magnitude, sorting their corresponding eigenvectors in te same order
  //! \param eigenvalues column vector containing eigenvalues
  //! \param eigenvectors Matrix containing the eigenvectors in its columns.
  //! It must have the same number of columns as <code>eigenvalues</code> has rows
  //! \return
  static pair<Matrix, Matrix> eigsort(Matrix eigenvalues, Matrix eigenvectors) {
//    if (eigenvalues.mCols != eigenvectors.mRows)
//      throw runtime_error("Incompatible number of eigenvalues and eigenvectors");

    Matrix eigval(eigenvalues.mRows, eigenvalues.mCols, eigenvalues.mData);
    Matrix eigvec(eigenvectors.mRows, eigenvectors.mCols, eigenvectors.mData);

    // keep the order of eigenvalues in this vector
    vector<size_t> newOrder;
    for (size_t i = 0; i < eigenvalues.nRows(); i++) {
      int position = 0;
      for (int j = 0; j < newOrder.size(); j++)
        if (eigenvalues(i, 0) < eigenvalues(newOrder[j], 0))
          position++;
      newOrder.insert(newOrder.begin() + position, i);
    }

    // order eigenvalues and eigenvectors by the value of the eigenvalues
    for (int i = 0; i < newOrder.size(); i++) {
      eigval(i, 0) = eigenvalues(newOrder[i], 0);

      for (int j = 0; j < eigenvectors.nRows(); j++) {
        eigvec(j, i) = eigenvectors(j, newOrder[i]);
      }
    }

    return make_pair(eigval, eigvec);
  }

 public:
  size_t nCols() { return mCols; }

  size_t nRows() { return mRows; }

  //region Constructors

  //! Initializes an empty matrix
  Matrix() {
    mRows = mCols = 0;
  }

  //! Initializes a matrix with a predetermined number of rows and columns
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  Matrix(size_t rows, size_t cols)
      : mRows(rows),
        mCols(cols),
        mData(rows * cols) {
  }

  //! Initializes a matrix with a predetermined number of rows and columns and populates it with data
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \param data a vector containing <code>rows * cols</code> elements to populate the matrix
  Matrix(size_t rows, size_t cols, const vector<double> &data)
      : mRows(rows),
        mCols(cols) {
    if (data.size() != rows * cols)
      throw runtime_error("Matrix dimension incompatible with its initializing vector.");
    mData = data;
  }
  //endregion

  //region Operators

  //region Scalar operators

  //! Scalar addition
  //! \param m a matrix
  //! \param value scalar to be added to the matrix
  //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
  friend Matrix operator+(const Matrix &m, double value) {
    Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = value + m(i, j);
      }
    }

    return result;
  }

  //! Scalar addition
  //! \param m a matrix
  //! \param value scalar to be added to the matrix
  //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
  friend Matrix operator+(double value, const Matrix &m) {
    return m + value;
  }

  //! Scalar subtraction
  //! \param m a matrix
  //! \param value scalar to be subtracted to the matrix
  //! \return the result of the scalar subtraction of <code>m</code> and <code>value</code>
  friend Matrix operator-(const Matrix &m, double value) {
    return m + (-value);
  }

  //! Scalar subtraction
  //! \param m a matrix
  //! \param value scalar to be subtracted to the matrix
  //! \return the result of the scalar subtraction of <code>m</code> and <code>value</code>
  friend Matrix operator-(double value, const Matrix &m) {
    return m - value;
  }

  //! Scalar multiplication
  //! \param m a matrix
  //! \param value scalar to be multiplied by the matrix
  //! \return the result of the scalar multiplication of <code>m</code> and <code>value</code>
  friend Matrix operator*(const Matrix &m, double value) {
    Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = value * m(i, j);
      }
    }

    return result;
  }

  //! Scalar multiplication
  //! \param m a matrix
  //! \param value scalar to be multiplied by the matrix
  //! \return the result of the scalar multiplication of <code>m</code> and <code>value</code>
  friend Matrix operator*(double value, const Matrix &m) {
    return m * value;
  }

  //! Scalar division
  //! \param m a matrix
  //! \param value scalar to be divide the matrix by
  //! \return the result of the scalar division of <code>m</code> by <code>value</code>
  friend Matrix operator/(const Matrix &m, double value) {
    Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = m(i, j) / value;
      }
    }

    return result;
  }

  //! Scalar division
  //! \param value scalar that will be divided by the matrix
  //! \param m a matrix
  //! \return the result of the scalar division of <code>value</code> by <code>m</code>
  friend Matrix operator/(double value, const Matrix &m) {
    // division is not commutative, so a new method is implemented
    Matrix result(m.mRows, m.mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m.mRows; i++) {
      for (size_t j = 0; j < m.mCols; j++) {
        result(i, j) = value / m(i, j);
      }
    }

    return result;
  }

  Matrix operator+=(double value) {
#pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] += value;
    return *this;
  }

  Matrix operator-=(double value) {
#pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] -= value;
    return *this;
  }

  Matrix operator*=(double value) {
#pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] *= value;
    return *this;
  }

  Matrix operator/=(double value) {
#pragma omp parallel for
    for (int i = 0; i < mData.size(); i++)
      mData[i] /= value;
    return *this;
  }
  //endregion

  //region Matrix operators

  //! Matrix addition operation
  //! \param b another matrix
  //! \return Result of the addition of both matrices
  Matrix operator+(const Matrix &b) {
    if (mRows != b.mRows || mCols != b.mCols)
      throw runtime_error("Cannot add these matrices");

    Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = this->operator()(i, j) + b(i, j);
      }
    }

    return result;
  }

  //! Matrix subtraction operation
  //! \param b another matrix
  //! \return Result of the subtraction of both matrices
  Matrix operator-(const Matrix &b) {
    if (mRows != b.mRows || mCols != b.mCols)
      throw runtime_error("Cannot add these matrices");

    Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = this->operator()(i, j) - b(i, j);
      }
    }

    return result;
  }

  //! Matrix multiplication operation
  //! \param b another matrix
  //! \return Result of the multiplication of both matrices
  Matrix operator*(const Matrix &b) {
    if (mCols != b.mRows)
      throw runtime_error(
          "Cannot multiply these matrices: left hand " + to_string(this->mRows) + "x" +
              to_string(this->mCols) + ", right hand " + to_string(b.mRows) + "x" + to_string(b.mCols));

    Matrix result(mRows, b.mCols);

#pragma omp parallel for collapse(2)
    // two loops iterate through every cell of the new matrix
    for (size_t i = 0; i < result.mRows; i++) {
      for (size_t j = 0; j < result.mCols; j++) {
        // here we calculate the value of a single cell in our new matrix
        result(i, j) = 0;
        for (size_t ii = 0; ii < mCols; ii++)
          result(i, j) += this->operator()(i, ii) * b(ii, j);
      }
    }
    return result;
  }
  Matrix &operator+=(const Matrix &other) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < other.mRows; i++) {
      for (size_t j = 0; j < other.mCols; j++) {
        this->operator()(i, j) += other(i, j);
      }
    }

    return *this;
  }

  Matrix &operator-=(const Matrix &other) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < other.mRows; i++) {
      for (size_t j = 0; j < other.mCols; j++) {
        this->operator()(i, j) -= other(i, j);
      }
    }

    return *this;
  }

  Matrix &operator*=(const Matrix &other) {
    if (mCols != other.mRows)
      throw runtime_error(
          "Cannot multiply these matrices: left hand " + to_string(this->mRows) + "x" +
              to_string(this->mCols) + ", right hand " + to_string(other.mRows) + "x" + to_string(other.mCols));

    Matrix result(mRows, other.mCols);

#pragma omp parallel for collapse(2)
    // two loops iterate through every cell of the new matrix
    for (size_t i = 0; i < result.mRows; i++) {
      for (size_t j = 0; j < result.mCols; j++) {
        // here we calculate the value of a single cell in our new matrix
        result(i, j) = 0;
        for (size_t ii = 0; ii < mCols; ii++)
          result(i, j) += this->operator()(i, ii) * other(ii, j);
      }
    }

    mRows = result.mRows;
    mCols = result.mCols;
    mData = result.mData;
    return *this;
  }
  //endregion

  //region Equality operators

  Matrix operator==(const double &value) {
    Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = this->operator()(i, j) == value;
      }
    }

    return result;
  }

  bool operator==(const Matrix &other) {
    if (mData.size() != other.mData.size() || mRows != other.mRows || mCols != other.mCols)
      return false;

    for (int k = 0; k < mData.size(); k++) {
      if (mData[k] != other.mData[k])return false;
    }

    return true;
  }

  Matrix operator!=(const double &value) {
    // subtract 1 from everything: 0s become -1s, 1s become 0s
    // negate everything: 0s remains 0s, -1s becomes 1s
    return -((*this == value) - 1);
  }
  //endregion

  //! Matrix negative operation
  //! \return The negative of the current matrix
  Matrix operator-() {
    Matrix result(this->mRows, this->mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mCols; i++) {
      for (size_t j = 0; j < mRows; j++) {
        result(i, j) = -this->operator()(i, j);
      }
    }

    return result;
  }

  //region Functors

  //! Functor used to access elements in the matrix
  //! \param i row index
  //! \param j column index
  //! \return element in position ij of the matrix
  double &operator()(size_t i, size_t j) {
    validateIndexes(i, j);
    return mData[i * mCols + j];
  }

  //! Functor used to access elements in the matrix
  //! \param i row index
  //! \param j column index
  //! \return element in position ij of the matrix
  double operator()(size_t i, size_t j) const {
    validateIndexes(i, j);
    return mData[i * mCols + j];
  }
  //endregion
  //endregion

  //! Returns a matrix filled with a single value
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \param value value to be used for initialization
  //! \return a matrix with all values set to <code>value</code>
  static Matrix fill(size_t rows, size_t cols, double value) {
    Matrix result(rows, cols, vector<double>(rows * cols, value));
    return result;
  }

  //! Creates a square matrix with a fixed value on the diagonal
  //! \param size dimensions of the square matrix
  //! \param value value to be used in the diagonal
  //! \return square matrix with a fixed value on the diagonal
  static Matrix diagonal(size_t size, double value) {
    Matrix result = zeros(size, size);
    for (size_t i = 0; i < size; i++)
      result(i, i) = value;

    return result;
  }

  bool isSquare() const {
    return mCols == mRows;
  }

  //! \return diagonal of the square matrix as a column vector
  Matrix diagonal() {
    if (!isSquare()) {
      throw runtime_error("Can't get the diagonal, not a square matrix");
    }

    Matrix result(mRows, 1);

#pragma omp parallel
    for (size_t i = 0; i < mRows; i++)
      result(i, 0) = this->operator()(i, i);

    return result;
  }

  //! Returns the identity matrix
  //! \param size dimensions of the square matrix
  //! \return identity matrix
  static Matrix identity(size_t size) {
    return diagonal(size, 1);
  }

  //! Returns a matrix filled with ones
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \return matrix filled with ones
  static Matrix ones(size_t rows, size_t cols) {
    return fill(rows, cols, 1);
  }

  //! Returns a matrix filled with zeros
  //! \param rows number of rows in the matrix
  //! \param cols number of columns in the matrix
  //! \return matrix filled with zeros
  static Matrix zeros(size_t rows, size_t cols) {
    return fill(rows, cols, 0);
  }

  //! Executes the Hadamard, or entrywise multiplication between two matrices
  //! \param b The other matrix
  //! \return result of the Hadamard multiplication of the two matrices
  Matrix hadamard(const Matrix &b) {
    if (mCols != b.mCols || mRows != b.mRows)
      throw runtime_error("Matrices have different dimentions");

    Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = this->operator()(i, j) * b(i, j);
      }
    }

    return result;
  }

  //! Returns a submatrix of the current matrix, removing one row and column of the original matrix
  //! \param row index of the row to be removed
  //! \param column index of the column to be removed
  //! \return submatrix of the current matrix, with one less row and column
  Matrix submatrix(size_t row, size_t column) const {
    Matrix result(mRows - 1, mCols - 1);

    size_t subi = 0;

#pragma omp parallel for
    for (size_t i = 0; i < mRows; i++) {
      size_t subj = 0;
      if (i == row) continue;
      for (size_t j = 0; j < mCols; j++) {
        if (j == column) continue;
        result(subi, subj) = this->operator()(i, j);
        subj++;
      }
      subi++;
    }

    return result;
  }

  //! Returns the minor of a matrix, which is the determinant of a submatrix
  //! where a single row and column are removed
  //! \param row index of the row to be removed
  //! \param column index of the column to be removed
  //! \return minor of the current matrix
  double getMinor(size_t row, size_t column) const {
//        the minor of a 2x2 a b is d c
//                           c d    b a
    if (mRows == 2 and mCols == 2) {
      Matrix result(2, 2);
      result(0, 0) = this->operator()(1, 1);
      result(0, 1) = this->operator()(1, 0);
      result(1, 0) = this->operator()(0, 1);
      result(1, 1) = this->operator()(0, 0);
      return result.determinant();
    }

    return submatrix(row, column).determinant();
  }

  //! Calculates the cofactor of a matrix at a given point
  //! \param row index of the row where the cofactor will be calculated
  //! \param column index of the column where the cofactor will be calculated
  //! \return cofactor of the matrix at the given position
  double cofactor(size_t row, size_t column) const {
    double minor;

    // special case for when our matrix is 2x2
    if (mRows == 2 and mCols == 2) {
      if (row == 0 and column == 0)
        minor = this->operator()(1, 1);
      else if (row == 1 and column == 1)
        minor = this->operator()(0, 0);
      else if (row == 0 and column == 1)
        minor = this->operator()(1, 0);
      else if (row == 1 and column == 0)
        minor = this->operator()(0, 1);
    } else
      minor = this->getMinor(row, column);
    return (row + column) % 2 == 0 ? minor : -minor;
  }

  //! Calculates the cofactor matrix
  //! \return Cofactor matrix of the current matrix
  Matrix cofactorMatrix() const {
    Matrix result(mRows, mCols);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = cofactor(i, j);
      }
    }
    return result;
  }

  //! Returns the adjugate of the current matrix, which is the transpose of its cofactor matrix
  //! \return Adjugate of the current matrix
  Matrix adjugate() const {
    return cofactorMatrix().transpose();
  }

  //! Calculates the inverse of the current matrix. Raises an error if
  //! the matrix is singular, that is, its determinant is equal to 0
  //! \return inverse of the current matrix
  Matrix inverse() const {
    if (!isSquare())
      throw runtime_error("Cannot invert a non-square matrix");

    double det = determinant();

    if (det == 0)
      throw runtime_error("Matrix is singular");

    Matrix adj = adjugate();
    return adjugate() / det;
  };

  //! Calculates the determinant of the matrix
  //! \return determinant of the matrix
  double determinant() const {
    if (!isSquare()) {
      throw runtime_error("Cannot calculate the determinant of a non-square matrix");
    }

    size_t n = mRows;
    double d = 0;
    if (n == 2) {
      return ((this->operator()(0, 0) * this->operator()(1, 1)) -
          (this->operator()(1, 0) * this->operator()(0, 1)));
    } else {
#pragma omp parallel for reduction (+:d)
      for (size_t c = 0; c < n; c++) {
        d += pow(-1, c) * this->operator()(0, c) * submatrix(0, c).determinant();
      }
      return d;
    }
  }

  //! Returns the transpose of a matrix
  //! \return transpose of the current matrix
  Matrix transpose() const {
    Matrix result(mCols, mRows);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(j, i) = this->operator()(i, j);
      }
    }

    return result;
  }

  //! Adds a column at the end of the matrix. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new column
  void addColumn(Matrix values) {
    addColumn(values, mCols);
  }

  //! Adds a row at the end of the matrix. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new row
  void addRow(Matrix values) {
    addRow(values, mRows);
  }

  //! Adds a column to the matrix at the given position. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new column
  //! \param position index of the new column. The column at the current
  //! position and all columns succeeding it are pushed forward.
  void addColumn(Matrix values, size_t position) {
    if (!isEmpty() and values.nRows() != mRows)
      throw runtime_error("Wrong number of values passed for new column");
    if (values.nCols() != 1)
      throw runtime_error("Can't add multiple columns at once");

    if (isEmpty()) {
      mRows = values.mRows;
      mCols = values.mCols;
      mData = values.mData;
      return;
    }

    // this is how you stop a reverse for loop with unsigned integers
    for (size_t i = mRows - 1; i != (size_t) -1; i--)
      mData.insert(mData.begin() + (i * mCols + position), values(i, 0));

    mCols += 1;
  }

  //! Adds a row to the matrix at the given position. Addition is done inplace.
  //! \param values a column vector containing the values to be added in the new row
  //! \param index index of the new row. The row at the current
  //! position and all rows succeeding it are pushed forward.
  void addRow(Matrix values, size_t index) {
    if (!isEmpty() and values.mRows != mCols)
      throw runtime_error("Wrong number of values passed for new row");
    if (values.mCols != 1)
      throw runtime_error("Can't add multiple rows at once");

    if (isEmpty()) {
      mRows = values.mCols;
      mCols = values.mRows;
      mData = values.mData;
      return;
    }

    for (size_t i = 0; i != mCols; i++) {
      size_t exact_position = index * (mCols) + i;
      if (exact_position < mData.size())
        mData.insert(mData.begin() + exact_position, values(i, 0));
      else
        mData.push_back(values(i, 0));
    }

    mRows += 1;
  }

  //! Removes a column from the matrix. Removal is done inplace.
  //! \param position index of the column to be removed.
  void removeColumn(int position) {
    // this is how you stop a reverse for loop with unsigned integers
    for (size_t i = mRows - 1; i != (size_t) -1; i--)
      mData.erase(mData.begin() + (i * mCols + position));

    mCols -= 1;
  }

  //! Returns only unique values from the matrix
  //! \return column vector containing the unique values from the matrix
  Matrix unique() const {
    // include all data from the inner vector in a set
    set<double> s;
    vector<double> auxVec;
    unsigned long size = mData.size();

    for (unsigned i = 0; i < size; ++i)
      s.insert(mData[i]);

    // include all the data from the set back into a vector
    auxVec.assign(s.begin(), s.end());

    // return a column matrix with the unique elements
    return Matrix(auxVec.size(), 1, auxVec);
  }

  //! Sorts elements of the matrix inplace
  void sort() {
    // just sort the inner vector
    std::sort(mData.begin(), mData.end());
  }

  //! Sorts the elements of a matrix
  //! \param m the matrix whose elements will be sorted
  //! \return a matrix with the same shape as <code>m</code>, with its elements sorted
  static Matrix sort(Matrix m) {
    // copy the inner vector of the matrix passed as argument
    // and return a new matrix with the sorted inner vector
    vector<double> data = m.mData;
    std::sort(data.begin(), data.end());
    return Matrix(m.mRows, m.mCols, data);
  }

  //! Counts occurrences of elements in a matrix.
  //! \return matrix with two columns. The first contains unique instances of the elements in the original matrix.
  //! The second column contains occurrences of the elements in the first column.
  Matrix count() {
    Matrix result = unique();
    result.sort();

    result.addColumn(zeros(result.mRows, 1), 1);

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++)
      for (size_t j = 0; j < mCols; j++)
        for (size_t g = 0; g < result.mRows; g++)
          if (this->operator()(i, j) == result(g, 0)) {
            result(g, 1)++;
            break;
          }

    return result;
  }

  //! Calculates means of a matrix, grouped by classes
  //! \param groups a column vector containing group assignments
  //! \return a matrix containing as many columns as there are unique groups.
  //! Each column represents the mean of each group.
  //! Columns are sorted by the group numbers in ascending order.
  Matrix mean(Matrix groups) {
    if (mRows != groups.mRows)
      throw runtime_error("Not enough groups for every element in the matrix");

    Matrix groupCount = groups.count();
    Matrix result = zeros(groupCount.mRows, mCols);

#pragma omp parallel for
    for (size_t i = 0; i < mRows; i++) {
      for (size_t g = 0; g < groupCount.mRows; g++) {
        if (groups(i, 0) == groupCount(g, 0)) {
          for (size_t j = 0; j < mCols; j++) {
            result(g, j) += this->operator()(i, j);
          }
          break;
        }
      }
    }

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < result.mRows; i++)
      for (size_t j = 0; j < result.mCols; j++)
        result(i, j) /= groupCount(i, 1);

    return result;
  }

  //! Calculates the mean of the columns of the matrix
  //! \return Column vector containing the means
  Matrix mean() {
    Matrix result = zeros(mCols, 1);

    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(j, 0) += this->operator()(i, j);
      }
    }

    result /= mRows;

    return result;
  }

  //! Calculates the scatter matrix. Columns are taken as features.
  //! \return scatter matrix
  Matrix scatter() {
    Matrix means = mean();
    Matrix result(mCols, mCols);

    for (size_t i = 0; i < mRows; i++) {
      Matrix rowDiff = getRow(i) - means;
      result += rowDiff * rowDiff.transpose();
    }

    return result;
  }

  //! Calculates the covariance matrix of the current matrix. Columns are taken as features.
  //! \return covariance matrix
  Matrix cov() {
    return scatter() / (mRows - 1);
  }

  //! Calculates the variance of the columns of the matrix
  //! \return Column vector containing the variances
  Matrix var() {
    Matrix means = mean();
    Matrix result = zeros(mCols, 1);

    for (size_t i = 0; i < mCols; i++) {
      for (size_t ii = 0; ii < mRows; ii++)
        result(i, 0) += pow((this->operator()(ii, i) - means(i, 0)), 2);

      result(i, 0) /= (mRows - 1);
    }

    return result;
  }

  //! Calculates the standard deviation of the columns of the matrix
  //! \return column vector containing standard deviations
  Matrix stdev() {
    Matrix result = var();

#pragma omp parallel for
    for (size_t i = 0; i < mCols; i++)
      result(i, 0) = sqrt(result(i, 0));

    return result;
  }

  //! Reshapes the current matrix. The operation is done inplace.
  //! \param rows new number of rows
  //! \param cols new number of columns
  void reshape(size_t rows, size_t cols) {
    if (mData.size() != rows * cols)
      throw runtime_error(
          "Invalid shape (" + to_string(rows) + "x" +
              to_string(cols) + " = " + to_string(rows * cols) +
              ") for a matrix with" + to_string(mData.size()) + " elements");

    mRows = rows;
    mCols = cols;
  }

  //! Gets a column from the matrix
  //! \param index index of the desired column
  //! \return column vector containing the values in the given column of the original matrix
  Matrix getColumn(size_t index) {
    if (index >= mCols)
      throw runtime_error("Column index out of bounds");

    Matrix result(mRows, 1);
#pragma omp parallel for
    for (size_t i = 0; i < mRows; i++)
      result(i, 0) = this->operator()(i, index);

    return result;
  }

  //! Gets a row from the matrix
  //! \param index index of the desired row
  //! \return column vector containing the values in the given row of the original matrix
  Matrix getRow(size_t index) {
    if (index >= mRows)
      throw runtime_error("Row index out of bounds");

    Matrix result(mCols, 1);
#pragma omp parallel for
    for (size_t i = 0; i < mCols; i++)
      result(i, 0) = this->operator()(index, i);

    return result;
  }

  //! Prints a matrix
  //! \param os output stream
  //! \param matrix the matrix to be printed
  //! \return output stream with the string representation of the matrix
  friend ostream &operator<<(ostream &os, const Matrix &matrix) {
    const int numWidth = 13;
    char fill = ' ';

    for (int i = 0; i < matrix.mRows; i++) {
      for (int j = 0; j < matrix.mCols; j++) {
        // the trick to print a table-like structure was stolen from here
        // https://stackoverflow.com/a/14796892
        os << left << setw(numWidth) << setfill(fill) << to_string(matrix(i, j));
      }
      os << endl;
    }

    return os;
  }

  //! Loads CSV data into a matrix
  //! \param path path of the CSV file
  //! \return a matrix constructed from the contents of the CSV file
  static Matrix fromCSV(const string &path) {
    vector<vector<double>> outer;
    vector<double> innerVector;

    ifstream arquivo(path);
    if (!arquivo.good())
      throw runtime_error("File '" + path + "' doesn't exist");

    unsigned long numCols = 0;

    while (!(innerVector = getNextLineAndSplitIntoTokens(arquivo)).empty()) {
      if (numCols == 0)
        numCols = innerVector.size();
      else if (numCols != innerVector.size())
        throw runtime_error("File has missing values in some columns");
      outer.push_back(innerVector);
    }

    Matrix result(outer.size(), numCols);

    for (size_t i = 0; i < result.mRows; i++)
      for (size_t j = 0; j < result.mCols; j++)
        result(i, j) = outer[i][j];

    return result;
  }

  //! Creates a diagonal matrix from a row or column vector
  //! \return diagonal matrix generated from the vector
  Matrix asDiagonal() {
    if (mRows != 1 and mCols != 1)
      throw runtime_error("Can't diagonalize, not a vector");

    size_t dimension = mCols > 1 ? mCols : mRows;

    Matrix result = zeros(dimension, dimension);

#pragma omp parallel for
    for (size_t i = 0; i < dimension; i++) {
      result(i, i) = mCols > 1 ? this->operator()(0, i) : this->operator()(i, 0);
    }
    return result;
  }

  //! Returns a copy of the matrix
  //! \return copy of the current matrix
  Matrix copy() {
    Matrix result(mRows, mCols);
    result.mData = mData;
    return result;
  }

  //! Standardizes the columns of the matrix, subtracting each element of a column
  //! by the column mean and dividing it by the standard deviation of the column.
  //! \return a new matrix with the columns standardized as described
  Matrix standardize() {
    Matrix result = copy(), means = mean(), stds = stdev();

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = (this->operator()(i, j) - means(j, 0)) / stds(j, 0);
      }
    }

    return result;
  }

  Matrix minusMean() {
    Matrix result = copy(), means = mean();

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        result(i, j) = this->operator()(i, j) - means(j, 0);
      }
    }

    return result;
  }

  //! Checks if the matrix contains a value
  //! \param value the vaue to look for
  //! \return true if the matrix contains the value, otherwise false
  bool contains(double value) {
    return std::find(mData.begin(), mData.end(), value) != mData.end();
  }

  //! Checks if the matrix is empty or uninitialized
  //! \return true if the matrix is empty or uninitialized, otherwise false
  bool isEmpty() {
    return mCols == 0 and mRows == 0;
  }

  //! Selects a subset of either columns or rows of the matrix
  //! \param bin a column vector containing only 0s and 1s, where indices with
  //! 1s indicate the indices of the columns/rows that will be returned by the method
  //! \param columns true if the filter will select the columns of the matrix, otherwise, rows will be selected
  //! \return
  Matrix filter(const Matrix bin, bool columns = false) {
    size_t dimension = columns ? mCols : mRows;

    if (bin.mCols != 1)
      throw runtime_error("Binary filter must have only one column");
    if (bin.mRows != dimension)
      throw runtime_error("Binary filter has the wrong number of row entries");

    Matrix uniqueBin = bin.unique();
    if (uniqueBin.mRows != 2 or !(uniqueBin.contains(1) and uniqueBin.contains(0)))
      throw runtime_error("Binary filter must be composed of only 0s and 1s");

    Matrix result;

    for (size_t i = 0; i < bin.mRows; i++) {
      if (bin(i, 0) == 1) {
        if (columns)
          result.addColumn(getColumn(i));
        else
          result.addRow(getRow(i));
      }
    }

    return result;
  }

  //! Selects a subset of rows of the matrix
  //! \param bin a column vector containing only 0s and 1s, where indices with
  //! 1s indicate the indices of the columns/rows that will be returned by the method
  Matrix getRows(const Matrix bin) {
    return filter(bin);
  }

  //! Selects a subset of columns of the matrix
  //! \param bin a column vector containing only 0s and 1s, where indices with
  //! 1s indicate the indices of the columns/rows that will be returned by the method
  Matrix getColumns(const Matrix bin) {
    return filter(bin, true);
  }

  //! Checks if the matrix is symmetric. A matrix is symmetric if it is equal to its transpose
  //! \return true if it is symmetric, otherwise false
  bool isSymmetric() {
    return *this == transpose();
  }

  //! Normalizes the column vectors of the matrix. Normalization is done by dividing
  //! each element of a vector by the length of the vector.
  //! \return Matrix with each column normalized by its length.
  Matrix normalize() {
    Matrix result(mRows, mCols, mData);

    // Calculate length of the column vector
    for (size_t j = 0; j < mCols; j++) {
      double length = 0;
#pragma omp parallel for reduction(+:length)
      for (size_t i = 0; i < mRows; i++) {
        length += pow(result(i, j), 2);
      }
      length = sqrt(length);

      // divide each element of the column by its length
#pragma omp parallel for
      for (size_t i = 0; i < mRows; i++) {
        result(i, j) /= length;
      }
    }

    return result;
  }

  //! Calculates the eigenvalues and eigenvectors of a matrix
  //! \param sort whether to sort the eigenvalues and eigenvectors
  //! \return a pair containing eigenvalues in a column vector in the first element of the pair
  //! and the correponding eigenvectors in the second element
  pair<Matrix, Matrix> eigen() {
    return isSymmetric() ? eigenSymmetric() : eigenNonSymmetric();
  };

  //! Calculates the eigenvalues and eigenvectors of a symmetric matrix using the Jacobi eigenvalue algorithm
  //! \return a pair containing eigenvalues in a column vector in the first element of the pair
  //! and the correponding eigenvectors in the second element
  pair<Matrix, Matrix> eigenSymmetric() {
    // Jacobi eigenvalue algorithm as explained
    // by profs Marina Andretta and Franklina Toledo

    // copy the current matrix
    Matrix A = copy();
    // initialize the eigenvector matrix
    Matrix V = identity(A.mCols);

    // get the tolerance
    double eps = numeric_limits<double>::epsilon();
    unsigned iterations = 0;

    // initiate the loop for numerical approximation of the eigenvalues
    while (true) {
      // find the element in the matrix with the largest modulo
      size_t p, q;
      double largest = 0;
      for (size_t i = 0; i < A.mRows; i++) {
        for (size_t j = 0; j < A.mCols; j++) {
          // it can't be in the diagonal
          if (i != j and abs(A(i, j)) > largest) {
            largest = abs(A(i, j));
            p = i;
            q = j;
          }
        }
      }

      // if the largest non-diagonal element of A is zero +/- eps,
      // it means A is almost diagonalized and the eigenvalues are
      // in the diagonal of A
      if (largest < 2 * eps or iterations >= 1000) {
        //eigenvalues are returned in a column matrix for convenience
        return eigsort(A.diagonal(), V);
      }

      iterations++;

      // else, perform a Jacobi rotation using this angle phi as reference
      double phi = (A(q, q) - A(p, p)) / (2 * A(p, q));
      double sign = (phi > 0) - (phi < 0);
      double t = phi == 0 ? 1 : 1 / (phi + sign * sqrt(pow(phi, 2) + 1));
      double cos = 1 / (sqrt(1 + pow(t, 2)));
      double sin = t / (sqrt(1 + pow(t, 2)));

      // the matrix that will apply the rotation is basically an identity matrix...
      Matrix U = identity(A.mRows);

      // ... with the exception of these values
      U(p, p) = U(q, q) = cos;
      U(p, q) = sin;
      U(q, p) = -sin;

      // apply the rotation
      A = U.transpose() * A * U;
      // update the corresponding eigenvectors
      V = V * U;
    }
  }

  //! Calculates the eigenvalues and eigenvectors of a on-symmetric matrix
  //! \param sort whether to sort the eigenvalues and eigenvectors
  //! \param calculate the eigenvalues and eigenvectors of a Hessenberg matrix with
  //! identical eigenvalues and eigenvectors instead
  //! \return a pair containing eigenvalues in a column vector in the first element of the pair
  //! and the correponding eigenvectors in the second element
  pair<Matrix, Matrix> eigenNonSymmetric(bool hes = true) {

    MatDoub unholyConvertion(static_cast<int>(mRows), static_cast<int>(mCols));

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < mRows; i++) {
      for (size_t j = 0; j < mCols; j++) {
        unholyConvertion[i][j] = this->operator()(i, j);
      }
    }

    Unsymmeig nr3Miracle(unholyConvertion, true, hes);

    Matrix eigenvalues(1, static_cast<size_t>(nr3Miracle.wri.size()));

    for (size_t i = 0; i < nr3Miracle.wri.size(); i++) {
      eigenvalues(0, i) = nr3Miracle.wri[i].real();
    }

    Matrix eigenvectors(static_cast<size_t>(nr3Miracle.zz.nrows()),
                        static_cast<size_t>(nr3Miracle.zz.ncols()));

    for (size_t i = 0; i < nr3Miracle.zz.nrows(); i++)
      for (size_t j = 0; j < nr3Miracle.zz.ncols(); j++)
        eigenvectors(i, j) = nr3Miracle.zz[i][j];

    return make_pair(eigenvalues, eigenvectors);
  }
};

#endif //MACHINE_LEARNING_MATRIX_HPP
