//
// Created by dodo on 11/10/17.
//

#ifndef MACHINE_LEARNING_LEASTSQUARES_HPP
#define MACHINE_LEARNING_LEASTSQUARES_HPP

#include <utility>
#include <vector>
#include "../include/matrix/Matrix.hpp"

using namespace std;


/**
 * Ordinary and weighted Least squares algorithm
 */
class LeastSquares {
public:
    enum RegressionType {
        REGULAR, WEIGHTED
    };
private:
    MatrixD X, y, coefs, residuals;
    RegressionType regressionType;
public:

    LeastSquares(MatrixD data, MatrixD labels, RegressionType regType = REGULAR) : regressionType(regType) {
        X = std::move(data);
        X.addColumn(MatrixD::ones(X.nRows(), 1), 0);
        y = std::move(labels);
    }

    RegressionType getRegressionType() const {
        return regressionType;
    }

    void setRegressionType(RegressionType regressionType) {
        this->regressionType = regressionType;
    }

    void fit() {
        // The formula for least squares is the following
        // B^ = (X'X)^{-1} X'y
        // Weighted least squares is like this
        // B^ = (X'WX)^{-1} X'Wy
        // where W is a Square matrix with the weights in the diagonal
        // if W = I, weighted least squares behaves just like ordinary least squares

        MatrixD W;

        if(regressionType == WEIGHTED) {
            MatrixD vars = X.transpose().var();
            W = vars.asDiagonal();
        } else
            W = MatrixD::identity(X.nRows());

        MatrixD Xt = X.transpose();
        MatrixD first_part = Xt * W * X;
        first_part = first_part.inverse();
        MatrixD second_part = Xt * W * y;
        coefs = first_part * second_part;

        residuals = y - (X * coefs);
        residuals = residuals.transpose() * residuals;
    }

    MatrixD predict(MatrixD m) {
        m.addColumn(MatrixD::ones(m.nRows(), 1), 0);
        return m * coefs;
    }

    const MatrixD &getCoefs() const {
        return coefs;
    }

    const MatrixD &getResiduals() const {
        return residuals;
    }
};


#endif // MACHINE_LEARNING_LEASTSQUARES_HPP
