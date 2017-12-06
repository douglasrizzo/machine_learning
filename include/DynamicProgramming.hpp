/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Class to solve the grid world toy problem using dynamic programming
 * @date   2017-12-04
 */


#ifndef MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
#define MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP

#include "Matrix.hpp"

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

    policy = MatrixD::ones(height * width, actions.size());
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

  void policyIteration() {

  }
};

#endif //MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
