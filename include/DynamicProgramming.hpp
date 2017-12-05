//
// Created by dodo on 04/12/17.
//

#ifndef MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
#define MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP

#include "Matrix.hpp"
class DynamicProgramming {
 private:
  MatrixD value, rewards, policy;
  vector<MatrixD> transition;
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

    transition = vector<MatrixD>(height * width);

    for (int i = 0; i < height * width; i++)
      transition[i] = MatrixD::zeros(height * width, actions.size());
  }
};

#endif //MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
