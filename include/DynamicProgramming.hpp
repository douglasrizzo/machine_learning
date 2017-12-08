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
  vector<pair<size_t, size_t>> goals;

  enum ActionType { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
  vector<ActionType> actions = {UP, DOWN, LEFT, RIGHT};

  /**
   * Transforms row x column coordinates from the grid world into a raster representation
   * @param row
   * @param col
   * @return
   */
  size_t fromCoord(size_t row, size_t col) {
    return row * value.nCols() + col;
  }

  /**
   * Transforms a raster coordinate from the grid world into its corresponding row x column representation
   * @param s
   * @return
   */
  pair<size_t, size_t> toCoord(size_t s) {
    size_t row = static_cast<size_t>(ceil((s + 1) / (double) value.nCols()) - 1);
    size_t col = s - row * value.nCols();

    return {row, col};
  }

 public:
  DynamicProgramming(size_t height, size_t width, vector<pair<size_t, size_t>> goals, double gamma = 1)
      : goals(goals), gamma(gamma) {

    // value function starts as 0 everywhere
    value = MatrixD::zeros(height, width);

    // set goal rewards are 0, all other states are -1
    rewards = MatrixD::fill(height, width, -1);
    for (auto goal:goals)
      rewards(goal.first, goal.second) = 0;

    // initialize the policy matrix giving equal probability of choice for every action
    policy = MatrixD::fill(height * width, actions.size(), 1.0 / actions.size());
  }

  bool isGoal(size_t s) {
    pair<size_t, size_t> stateCoords = toCoord(s);
    return std::find(goals.begin(), goals.end(), stateCoords) != goals.end();
  }

  double transition(size_t currentState, ActionType action, size_t nextState) {
    // the agent never leaves the goal
    if (isGoal(currentState))
      return 0;

    // return 1 if applying the given action actually takes the agent to the desired state
    size_t resultingState = applyAction(currentState, action);
    return resultingState == nextState;
  }

  size_t applyAction(size_t currentState, ActionType action) {
    pair<size_t, size_t> s1 = toCoord(currentState);
    size_t s1row = s1.first, s1col = s1.second, s2row, s2col;

    switch (action) {
      case UP:s2col = s1col;
        s2row = s1row != 0 ? s1row - 1 : s1row;
        break;
      case DOWN:s2col = s1col;
        s2row = s1row != value.nCols() - 1 ? s1row + 1 : s1row;
        break;
      case LEFT:s2row = s1row;
        s2col = s1col != 0 ? s1col - 1 : s1col;
        break;
      case RIGHT:s2row = s1row;
        s2col = s1col != value.nRows() - 1 ? s1col + 1 : s1col;
        break;
      default:return 0;
    }

    return fromCoord(s2row, s2col);
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

  size_t nextState(size_t s, ActionType a) {
    for (size_t ss = 0; ss < value.nRows() * value.nCols(); ss++) {
      if (transition(s, a, ss) == 1)
        return ss;
    }
    throw runtime_error("No next state found");
  }

  ActionType bestActionForState(size_t s) {
    ActionType best = UP;

    for (ActionType a:actions) {
      // TODO should consider if equal...
      if (actionValue(s, a) > actionValue(s, best))
        best = a;
    }

    return best;
  }

  double actionValue(size_t s, ActionType a) {
    double q = 0;

    for (size_t i = 0; i < value.nRows(); ++i) {
      for (size_t j = 0; j < value.nCols(); ++j) {
        size_t s2 = fromCoord(i, j);
        // WARNING: here the ActionType a is being used as an index
        q += transition(s, a, s2) * policy(s, a) * (rewards(i, j) + (gamma * value(i, j)));
      }
    }

    return q;
  }

  Matrix<double> bestPolicyForState(size_t s) {
    MatrixD result = MatrixD::zeros(1, actions.size());

    int bestIndex = 0;
    ActionType best = actions[bestIndex];
    for (int i = 1; i < actions.size(); i++) {
      if (actionValue(s, actions[i]) > actionValue(s, actions[bestIndex]))
        bestIndex = i;
    }

    result(0, bestIndex) = 1;
    return result;
  }

  ActionType actionFromPolicy(size_t s) {
    MatrixD statePolicy = policyForState(s);
    double prob = MersenneTwister().d_random();

    for (size_t i = 1; i < statePolicy.nRows(); i++)
      statePolicy(i, 0) += statePolicy(i - 1, 0);

    for (size_t i = 0; i < statePolicy.nRows() - 1; i++) {
      if (prob <= statePolicy(i, 0))
        return actions[i];
    }
    return actions[actions.size() - 1];
  }

  void policyIteration(double threshold = .001, bool verbose = true) {
    // step 1: initialization was done in the constructor
    // step 2: policy evaluation
    double delta = 0;

    do {
      for (size_t i = 0; i < value.nRows(); i++) {
        for (size_t j = 0; j < value.nCols(); j++) {
          size_t state1 = fromCoord(i, j);

          double currentV = value(i, j), newV = 0;

          for (ActionType action : actions)
            newV += actionValue(state1, action);

          value(i, j) = newV;

          double newDelta = abs(currentV - value(i, j));

          if (newDelta > delta)
            delta = newDelta;
        }
      }
      if (verbose) cout << value << endl;
    } while (delta >= threshold);


    // step 3: policy improvement
    MatrixD b;
    bool stablePolicy;
    do {
      stablePolicy = true;
      for (size_t i = 0; i < value.nRows(); ++i) {
        for (size_t j = 0; j < value.nCols(); ++j) {
          size_t state = fromCoord(i, j);
          b = policyForState(state);

          policy.setRow(state, bestPolicyForState(state));

          if (stablePolicy and b != policy.getRow(state))
            stablePolicy = false;
        }
      }
    } while (!stablePolicy);
  }

  void valueIteration(double threshold = .001, bool verbose = true) {
    // initialization was done in the constructor
    double delta = 0;

    do {
      for (size_t i = 0; i < value.nRows(); ++i) {
        for (size_t j = 0; j < value.nCols(); ++j) {
          size_t state1 = fromCoord(i, j);

          double currentV = value(i, j);
          value(i, j) = actionValue(state1, bestActionForState(state1));
          policy.setRow(state1, bestPolicyForState(state1));

          double newDelta = abs(currentV - value(i, j));

          if (newDelta > delta)
            delta = newDelta;
        }
      }
      if (verbose) cout << value << endl;
    } while (delta >= threshold);
  }

  void onPolicyMonteCarloControl(unsigned nIters) {
    for (unsigned iter = 0; iter < nIters; iter++) {

    }
  }
};

#endif //MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
