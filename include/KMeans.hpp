/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Implementation of the k-means algorithm
 * @date   2017-10-25
 */

#ifndef MACHINE_LEARNING_KMEANS_HPP
#define MACHINE_LEARNING_KMEANS_HPP

#include "../include/matrix/Matrix.hpp"
#include "Metrics.hpp"
#include "../include/mersenne_twister/MersenneTwister.hpp"


/**
 * Implementaion of the k-means algorithm
 */
class KMeans {
public:
    enum InitializationMethod { RANDOM, SAMPLE };
private:
    MatrixD X, y, centroids;
    unsigned int k, totalIterations;
    double distance, sse;
    InitializationMethod initMethod;

    /**
     * @return Sum of squared errors between elements and their centroids
     */
    double SSE() {
        return Metrics::euclidean(X, centroids, false).sum();
    }

public:

    KMeans() {}

    /**
     * Assigns elements of a data set to clusters
     * @param data a Matrix containing elements in rows and features in columns
     * @return column vector with the index of clusters each element is assigned to
     */
    MatrixD predict(MatrixD data) {
        if(centroids.nCols() != data.nCols())
            throw invalid_argument("Data elements and cluster centroids don't have the same number of dimensions.");

        MatrixD distances = Metrics::minkowski(data, centroids, distance, false);
        MatrixD results = MatrixD::zeros(data.nRows(), 1);

        for(size_t i = 0; i < distances.nRows(); i++) {
            for(size_t j = 1; j < distances.nCols(); j++) {
                double current_c = results(i, 0);
                double current_d = distances(i, current_c);
                double new_c = j;
                double new_d = distances(i, new_c);

                if(new_d < current_d)
                    results(i, 0) = new_c;
            }
        }

        return results;
    }

    /**
     * Find the k centroids that best fit the data.
     * @param data a Matrix containing the data
     * @param k number of clusters to be generated
     * @param iters number of maximum assignment/adjustment iterations
     * @param inits number of algorithm reinitialization
     * @param distance L norm of the distance measure to be used (1 for Manhattan, 2 for Euclidean etc.)
     * @param initMethod centroid initialization method
     * @param verbose whether to output progress or not
     */
    void fit(MatrixD data,
        unsigned int k,
        unsigned int iters = 100,
        unsigned int inits = 100,
        double distance = 2,
        InitializationMethod initMethod = SAMPLE, bool verbose = false) {
        this->X = data.standardize();
        this->k = k;
        this->initMethod = initMethod;
        this->distance = distance;
        this->totalIterations = 0;
        MersenneTwister twister;
        double minSSE;

        for(int currentInit = 0; currentInit < inits; currentInit++) {

            if(initMethod == RANDOM) {
                centroids = MatrixD(k, X.nCols());

                for(size_t i = 0; i < centroids.nRows(); i++)
                    for(size_t j = 0; j < centroids.nCols(); j++)
                        centroids(i, j) = twister.d_random(X.getColumn(j).min(),
                            X.getColumn(j).max());
            } else {
                vector<int> sample = twister.randomValues(X.nRows(), k, false);
                centroids = MatrixD();

                for(int i = 0; i < sample.size(); i++)
                    centroids.addRow(X.getRow(sample[i]));
            }

            MatrixD yPrev;

            for(int currentIteration = 0; currentIteration < iters; currentIteration++) {
                if(verbose)
                    cout << currentInit << '/' << inits + 1 << '\t'
                         << currentIteration << '/' << iters + 1 << '\t'
                         << SSE() << endl;

                MatrixD yCurr = predict(X); // assignment

                if(yCurr == yPrev)
                    break;

                yPrev = yCurr;
                centroids = X.mean(yCurr); // centroid update
            }

            if(currentInit == 0 or SSE() < minSSE) {
                minSSE = SSE();
                y = yPrev;
            }
        }

        this->sse = minSSE;
    }

    const MatrixD &getY() const {
        return y;
    }

    const MatrixD &getCentroids() const {
        return centroids;
    }

    unsigned int getK() const {
        return k;
    }

    unsigned int getTotalIterations() const {
        return totalIterations;
    }

    double getDistance() const {
        return distance;
    }

    double getSse() const {
        return sse;
    }
};


#endif // MACHINE_LEARNING_KMEANS_HPP
