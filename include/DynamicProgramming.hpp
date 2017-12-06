/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Class to solve the grid world toy problem using dynamic programming
 * @date   2017-12-04
 */


#ifndef MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
#define MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP

#include "Matrix.hpp"
#include "MersenneTwister.hpp"

class DynamicProgramming {
 private:
  MatrixD value, rewards, policy;
  double gamma;

  enum ActionType { UP, DOWN, LEFT, RIGHT };
  vector<ActionType> actions = {UP, DOWN, LEFT, RIGHT};

 public:
  DynamicProgramming(size_t height, size_t width, vector<pair<size_t, size_t>> goals)
      : gamma(gamma) {
    value = MatrixD::zeros(height, width);
    rewards = MatrixD::fill(height, width, -1);

    for (auto goal:goals)
      rewards(goal.first, goal.second) = 0;

    // initialize the policy matrix giving equal probability of choice for every action
    policy = MatrixD::fill(height * width, actions.size(), 1.0 / actions.size());
  }

  double transition(size_t currentState, ActionType action, size_t nextState) {
    switch (action) {
      case UP:return currentState - value.nCols() == nextState;
      case DOWN:return currentState + value.nCols() == nextState;
      case LEFT:return currentState - 1 % value.nCols() != value.nCols() - 1 && currentState - 1 == nextState;
      case RIGHT:return currentState - 1 % value.nCols() != 0 && currentState + 1 == nextState;
      default:return 0;
    }
  }

  MatrixD policyForState(size_t s) {
    MatrixD actionProportions = policy.getRow(s);
    double sum = actionProportions.sum();

    // normalize proportions so their sum is 1
    if (sum != 1)
      for (size_t i = 0; i < actionProportions.nRows(); i++)
        actionProportions(i, 0) /= sum;

    return actionProportions;
  }

  Matrix<double> bestActionForState(size_t s) {
    return Matrix<double>();
  }

  void policyIteration(double threshold = .001) {
    // step 1: initialization was done in the constructor
    // step 2: policy evaluation
    double delta = 0;
    for (size_t i = 0; i < value.nRows(); ++i) {
      for (size_t j = 0; j < value.nCols(); ++j) {
        double v = value(i, j);
        // TODO insane sum
        double newDelta = abs(v - value(i, j));

        if (newDelta > delta)
          delta = newDelta;
      }
    }

    // step 3: policy improvement
    MatrixD b;
    bool stablePolicy;
    do {
      stablePolicy = true;
      for (size_t i = 0; i < value.nRows(); ++i) {
        for (size_t j = 0; j < value.nCols(); ++j) {
          size_t state = i * policy.nRows() + j;
          b = policyForState(state);
          policy.setRow(state, bestActionForState(state));

          if (stablePolicy and b != policy.getRow(state))
            stablePolicy = false;
        }
      }
    } while (!stablePolicy);
  }
};

#endif //MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
