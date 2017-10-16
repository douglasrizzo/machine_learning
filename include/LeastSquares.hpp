//
// Created by dodo on 11/10/17.
//

#ifndef MACHINE_LEARNING_LEASTSQUARES_HPP
#define MACHINE_LEARNING_LEASTSQUARES_HPP

#include <utility>
#include <vector>
#include "Matrix.hpp"

using namespace std;

class LeastSquares {
public:
    enum RegressionType {
        REGULAR, WEIGHTED
    };
private:
    Matrix X, y, coefs, residuals;
    RegressionType regressionType;
public :
    LeastSquares(Matrix data, Matrix labels, RegressionType regType = REGULAR) : regressionType(regType) {
        X = std::move(data);
        X.addColumn(0, Matrix::ones(X.nRows(), 1));
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

        Matrix W;
        if (regressionType == WEIGHTED) {
            Matrix vars = X.transpose().var();
            W = vars.asDiagonal();
        } else
            W = Matrix::identity(X.nRows());

        Matrix Xt = X.transpose();
        Matrix first_part = Xt * W * X;
        first_part = first_part.inverse();
        Matrix second_part = Xt * W * y;
        coefs = first_part * second_part;

        residuals = y - (X * coefs);
        residuals= residuals.transpose()*residuals;
    }

    Matrix predict(Matrix m) {
        m.addColumn(0, Matrix::ones(1, X.nCols()));
        return m.transpose() * coefs;
    }

    const Matrix &getCoefs() const {
        return coefs;
    }

    const Matrix &getResiduals() const {
        return residuals;
    }
};

#endif //MACHINE_LEARNING_LEASTSQUARES_HPP
