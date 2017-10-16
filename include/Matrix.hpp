/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Matrix implementation, with a series of linear algebra functions
 * @date   2017-10-17
 */

#ifndef MACHINE_LEARNING_MATRIX_HPP
#define MACHINE_LEARNING_MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

//! Matrix implementation, with a series of linear algebra functions
class Matrix {
private:
    size_t mRows;
    size_t mCols;
    std::vector<double> mData;

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

public:
    size_t nCols() { return mCols; }

    size_t nRows() { return mRows; }

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

    //! Returns a matrix filled with a single value
    //! \param rows number of rows in the matrix
    //! \param cols number of columns in the matrix
    //! \param value value to be used for initialization
    //! \return a matrix with all values set to <code>value</code>
    static Matrix fill(size_t rows, size_t cols, double value) {
        Matrix result(rows, cols);

        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                result(i, j) = value;
            }
        }
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

    bool isSquare() {
        return mCols == mRows;
    }

    //! \return diagonal of the square matrix as a column vector
    Matrix diagonal() {
        if (!isSquare()) {
            throw runtime_error("Can't get the diagonal, not a square matrix");
        }
        Matrix result(mRows, 1);
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

    //! Scalar addition
    //! \param m a matrix
    //! \param value scalar to be added to the matrix
    //! \return the result of the scalar addition of <code>m</code> and <code>value</code>
    friend Matrix operator+(const Matrix &m, double value) {
        Matrix result(m.mRows, m.mCols);

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

        for (size_t i = 0; i < m.mRows; i++) {
            for (size_t j = 0; j < m.mCols; j++) {
                result(i, j) = value / m(i, j);
            }
        }

        return result;
    }

    //! Functor used to access elements in the matrix
    //! \param i row index
    //! \param j column index
    //! \return element in position ij of the matrix
    double &operator()(size_t i, size_t j) {
        return mData[i * mCols + j];
    }


    //! Functor used to access elements in the matrix
    //! \param i row index
    //! \param j column index
    //! \return element in position ij of the matrix
    double operator()(size_t i, size_t j) const {
        return mData[i * mCols + j];
    }

    //! Executes the Hadamard, or entrywise multiplication between two matrices
    //! \param b The other matrix
    //! \return result of the Hadamard multiplication of the two matrices
    Matrix hadamard(const Matrix &b) {
        if (mCols != b.mCols || mRows != b.mRows)
            throw runtime_error("Matrices have different dimentions");

        Matrix result(mRows, mCols);

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
    Matrix submatrix(size_t row, size_t column) {
        Matrix result(mRows - 1, mCols - 1);

        size_t subi = 0;
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
    double getMinor(size_t row, size_t column) {
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
    double cofactor(size_t row, size_t column) {
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
    Matrix cofactorMatrix() {
        Matrix result(mRows, mCols);
        for (size_t i = 0; i < mRows; i++) {
            for (size_t j = 0; j < mCols; j++) {
                result(i, j) = cofactor(i, j);
            }
        }
        return result;
    }

    //! Returns the adjugate of the current matrix, which is the transpose of its cofactor matrix
    //! \return Adjugate of the current matrix
    Matrix adjugate() {
        return cofactorMatrix().transpose();
    }

    //! Calculates the inverse of the current matrix. Raises an error if
    //! the matrix is singular, that is, its determinant is equal to 0
    //! \return inverse of the current matrix
    Matrix inverse() {
        if (!isSquare())
            throw runtime_error("Cannot invert a non-square matrix");

        double det = determinant();

        if (det == 0)
            throw runtime_error("Matrix is singular");

        Matrix adj = adjugate();
        return adjugate() / det;
    };

    double determinant() {
        if (!isSquare()) {
            throw runtime_error("Cannot calculate the determinant of a non-square matrix");
        }

        size_t n = mRows;
        double d = 0;
        if (n == 2) {
            return ((this->operator()(0, 0) * this->operator()(1, 1)) -
                    (this->operator()(1, 0) * this->operator()(0, 1)));
        } else {
            for (size_t c = 0; c < n; c++) {
                d += pow(-1, c) * this->operator()(0, c) * submatrix(0, c).determinant();
            }
            return d;
        }
    }

    //! Returns the transpose of a matrix
    //! \return transpose of the current matrix
    Matrix transpose() {
        Matrix result(mCols, mRows);

        for (size_t i = 0; i < mRows; i++) {
            for (size_t j = 0; j < mCols; j++) {
                result(j, i) = this->operator()(i, j);
            }
        }

        return result;
    }

    //! Adds a column the matrix at the given position. Addition is done inplace.
    //! \param position index of the new column. The column at the current
    //! position and all columns succeeding it are pushed forward.
    //! \param values a column vector containing the values to be added in the new column
    void addColumn(int position, Matrix values) {
        if (values.nRows() != mRows)
            throw runtime_error("Wrong number of values passed for new column");
        if (values.nCols() != 1)
            throw runtime_error("Can't add multiple columns at once");

        // this is how you stop a reverse for loop with unsigned integers
        for (size_t i = mRows - 1; i != (size_t) -1; i--)
            mData.insert(mData.begin() + (i * mCols + position), values(i, 0));

        mCols += 1;
    }

    //! Removes a column from the matrix. Removal is done inplace.
    //! \param position index of the column to be removed.
    void removeColumn(int position) {
        // this is how you stop a reverse for loop with unsigned integers
        for (size_t i = mRows - 1; i != (size_t) -1; i--)
            mData.erase(mData.begin() + (i * mCols + position));

        mCols -= 1;
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

        for (size_t i = 0; i < mCols; i++)
            result(i, 0) /= mRows;

        return result;
    }

    //! Calculates the covariance matrix of the current matrix. Columns are taken as features
    //! \return covariance matrix
    Matrix cov() {
        Matrix means = mean();

        Matrix result(mCols, mCols);

        for (size_t i = 0; i < mCols; i++) {
            for (size_t j = i; j < mCols; j++) {
                double cov = 0;

                for (size_t ii = 0; ii < mRows; ii++)
                    cov += (this->operator()(ii, i) - means(i, 0)) * (this->operator()(ii, j) - means(j, 0));

                result(i, j) = result(j, i) = cov / (mRows - 1);
            }
        }

        return result;
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
        for (size_t i = 0; i < mCols; i++)
            result(i, 0) = this->operator()(index, i);

        return result;
    }

    //! Matrix addition operation
    //! \param b another matrix
    //! \return Result of the addition of both matrices
    Matrix operator+(const Matrix &b) {
        if (mRows != b.mRows || mCols != b.mCols)
            throw runtime_error("Cannot add these matrices");

        Matrix result(mRows, mCols);

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


    //! Matrix negative operation
    //! \return The negative of the current matrix
    Matrix operator-() {
        Matrix result(this->mRows, this->mCols);

        for (size_t i = 0; i < mCols; i++) {
            for (size_t j = 0; j < mRows; j++) {
                result(i, j) = -this->operator()(i, j);
            }
        }

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

        for (size_t i = 0; i < dimension; i++) {
            result(i, i) = mCols > 1 ? this->operator()(0, i) : this->operator()(i, 0);
        }
        return result;
    }

    Matrix copy() {
        Matrix result(mRows, mCols);
        result.mData = mData;
        return result;
    }

    //! Calculates the eigenvalues and eigenvectors of the matrix using the Jacobi eigenvalue algorithm
    //! \return a pair containing eigenvalues in a column vector in the first element of the pair
    //! and the correponding eigenvectors in the second element
    pair<Matrix, Matrix> eigen() {
        // Jacobi eigenvalue algorithm as explained
        // by profs Marina Andretta and Franklina Toledo

        // copy the current matrix
        Matrix A = copy();
        // initialize the eigenvector matrix
        Matrix V = identity(A.mCols);

        // get the tolerance
        double eps = numeric_limits<double>::epsilon();

        // initiate the loop for numerical approximation of the eigenvalues
        while (true) {
            // find the element in the matrix with the largest modulo
            size_t p, q;
            double largest = 0;
            for (size_t i = 0; i < A.mRows; i++) {
                for (size_t j = 0; j < A.mCols; j++) {
                    // it can't be in the diagonal
                    if (i == j) continue;
                    if (abs(A(i, j)) > largest) {
                        largest = abs(A(i, j));
                        p = i;
                        q = j;
                    }
                }
            }

            // if the largest non-diagonal element of A is zero +/- eps,
            // it means A is almost diagonalized and the eigenvalues are
            // in the diagonal of A
            if (largest < 2 * eps) {
                //eigenvalues are returned in a column matrix for convenience
                return make_pair(A.diagonal(), V);
            }

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
};


#endif //MACHINE_LEARNING_MATRIX_HPP
