//
// Created by dodo on 31/10/17.
//

#ifndef MACHINE_LEARNING_METRICS_HPP
#define MACHINE_LEARNING_METRICS_HPP

#include "../include/matrix/Matrix.hpp"


/**
 * Distance metrics
 */
class Metrics {

public:

    // ! Calculates the Minkowski distances between elements in a matrix. Elements must be located in the matrix rows
    // ! \param m matrix of elements
    // ! \param p the power to be used by the metric
    // ! \param root whether to take the root of the distance. Default is False for the expected behavior
    // ! \return a matrix containing, in its n x m position, the distance between the nth and mth elements in the matrix
    static MatrixD minkowski(MatrixD m, double p, bool root = true) {
        MatrixD distances = MatrixD::zeros(m.nRows(), m.nRows());

        for(size_t i = 0; i < m.nRows(); i++) {
            for(size_t j = i + 1; j < m.nRows(); j++) {
                for(size_t k = 0; k < m.nCols(); k++) {
                    distances(i, j) += pow(abs(m(i, k) - m(j, k)), p);
                }

                if(root)
                    distances(i, j) = pow(distances(i, j), 1 / p);
            }
        }

        return distances;
    }

    // ! Calculates the Chebyshev distances between elements in a matrix. Elements must be located in the matrix rows
    // ! \param a matrix of elements
    // ! \return a matrix containing, in its n x m position, the distance between the nth and mth elements in the matrix
    static MatrixD chebyshev(MatrixD a) {
        MatrixD distances = MatrixD::zeros(a.nRows(), a.nRows());

        for(size_t i = 0; i < a.nRows(); i++) {
            for(size_t j = i + 1; j < a.nRows(); j++) {
                for(size_t k = 0; k < a.nCols(); k++) {
                    double x1 = a(i, k), x2 = a(j, k);
                    double term = abs(x1 - x2);
                    distances(i, j) = distances(j, i) = max(distances(i, j), term);
                }
            }
        }

        return distances;
    }

    // ! Calculates the Euclidean distances between elements in a matrix. Elements must be located in the matrix rows
    // ! \param m matrix of elements
    // ! \param root whether to take the root of the distance. Default is False for the expected behavior
    // ! \return a matrix containing, in its n x m position, the distance between the nth and mth elements in the matrix
    static MatrixD euclidean(MatrixD m, bool root = true) {
        return minkowski(m, 2, root);
    }

    // ! Calculates the Manhattan distances between elements in a matrix. Elements must be located in the matrix rows
    // ! \param m matrix of elements
    // ! \param root whether to take the root of the distance. Default is False for the expected behavior
    // ! \return a matrix containing, in its n x m position, the distance between the nth and mth elements in the matrix
    static MatrixD manhattan(MatrixD m, bool root = true) {
        return minkowski(m, 1, root);
    }

    // ! Calculates the Minkowski distances between elements in two matrices. Elements must be located in the matrices rows and both matrices must have the same number of features (columns)
    // ! \param a first matrix
    // ! \param b second matrix
    // ! \param p the power to be used by the metric
    // ! \param root whether to take the root of the distance. Default is False for the expected behavior
    // ! \return a matrix containing, in its n x m position, the distance between the nth element in the first matrix and the mth element in the second matrix
    static MatrixD minkowski(MatrixD a, MatrixD b, double p, bool root = true) {
        if(a.nCols() != b.nCols())
            throw runtime_error("Matrices have different number of dimensions");

        MatrixD distances = MatrixD::zeros(a.nRows(), b.nRows());

        for(size_t i = 0; i < a.nRows(); i++) {
            for(size_t j = 0; j < b.nRows(); j++) {
                for(size_t k = 0; k < a.nCols(); k++) {
                    double x1 = a(i, k), x2 = b(j, k);
                    double term = pow(abs(x1 - x2), p);
                    distances(i, j) += term;
                }

                if(root)

                    distances(i, j) = pow(distances(i, j), 1 / p);
            }
        }

        return distances;
    }

    // ! Calculates the Chebyshev distances between elements in two matrices. Elements must be located in the matrices rows and both matrices must have the same number of features (columns)
    // ! \param a first matrix
    // ! \param b second matrix
    // ! \return a matrix containing, in its n x m position, the distance between the nth element in the first matrix and the mth element in the second matrix
    static MatrixD chebyshev(MatrixD a, MatrixD b) {
        if(a.nCols() != b.nCols())
            throw runtime_error("Matrices have different number of dimensions");

        MatrixD distances = MatrixD::zeros(a.nRows(), b.nRows());

        for(size_t i = 0; i < a.nRows(); i++) {
            for(size_t j = 0; j < b.nRows(); j++) {
                for(size_t k = 0; k < a.nCols(); k++) {
                    double x1 = a(i, k), x2 = b(j, k);
                    double term = abs(x1 - x2);
                    distances(i, j) = max(distances(i, j), term);
                }
            }
        }

        return distances;
    }

    // ! Calculates the Euclidean distances between elements in two matrices. Elements must be located in the matrices rows and both matrices must have the same number of features (columns)
    // ! \param a first matrix
    // ! \param b second matrix
    // ! \param root whether to take the root of the distance. Default is False for the expected behavior
    // ! \return a matrix containing, in its n x m position, the distance between the nth element in the first matrix and the mth element in the second matrix
    static MatrixD euclidean(MatrixD a, MatrixD b, bool root = true) {
        return minkowski(a, b, 2, root);
    }

    // ! Calculates the Manhattan distances between elements in two matrices. Elements must be located in the matrices rows and both matrices must have the same number of features (columns)
    // ! \param a first matrix
    // ! \param b second matrix
    // ! \param root whether to take the root of the distance. Default is False for the expected behavior
    // ! \return a matrix containing, in its n x m position, the distance between the nth element in the first matrix and the mth element in the second matrix
    static MatrixD manhattan(MatrixD a, MatrixD b, bool root = true) {
        return minkowski(a, b, 1, root);
    }
};


#endif // MACHINE_LEARNING_METRICS_HPP
