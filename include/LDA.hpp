/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Linear discriminant analysis algorithm
 * @date   2017-10-18
 */

#ifndef MACHINE_LEARNING_LDA_HPP
#define MACHINE_LEARNING_LDA_HPP

#include <utility>

#include "../include/matrix/Matrix.hpp"

using namespace std;


/**
 * Linear discriminant analysis algorithm
 */
class LDA {
private:
    MatrixD X, y, eigenvalues, eigenvectors, transformedData;
public:

    /**
     * Linear discriminant analysis algorithm
     * @param data The matrix whose linear discriminants will be found
     * @param classes Column vector containing the classes each row element in <code>data</code> belongs to
     */
    LDA(MatrixD data, MatrixD classes) : X(std::move(data)), y(std::move(classes)) {
        if(data.nRows() != classes.nRows())
            throw invalid_argument("data and classes must have the same number of rows");

        if(classes.nCols() != 1)
            throw invalid_argument("classes must me a column vector");
    }

    void fit() {
        MatrixD Sw = X.WithinClassScatter(y);
        MatrixD Sb = X.BetweenClassScatter(y);

        auto eigen = (Sw.inverse() * Sb).eigen();

        eigenvalues = eigen.first;
        eigenvectors = eigen.second;

        transformedData = (eigenvectors.transpose() * X.transpose()).transpose();
    }

    /**
     * Transforms the data matrix using the eigenvectors found by <code>fit()</code>
     * @return
     */
    MatrixD transform() {
        return transformedData;
    }
};


#endif // MACHINE_LEARNING_LDA_HPP
