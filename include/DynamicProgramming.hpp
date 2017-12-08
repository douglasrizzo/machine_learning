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
    return normalizeToOne(policy.getRow(s));
  }

  size_t nextState(size_t s, ActionType a) {
    for (size_t ss = 0; ss < value.nRows() * value.nCols(); ss++) {
      if (transition(s, a, ss) == 1)
        return ss;
    }
    throw runtime_error("No next state found");
  }

  double actionValue(size_t s, ActionType a) {
    double q = 0;

    pair<size_t, size_t> coords = toCoord(s);
    double r = rewards(coords.first, coords.second);
    for (size_t i = 0; i < value.nRows(); ++i) {
      for (size_t j = 0; j < value.nCols(); ++j) {
        size_t s2 = fromCoord(i, j);
        double t = transition(s, a, s2);
        double v = value(i, j);

        // WARNING: here the ActionType a is being used as an index
        q += t * (r + gamma * v);

      }
    }

    return q;
  }

  Matrix<double> normalizeToOne(MatrixD m) {
    return m / m.sum();
  }

  vector<double> actionValuesForState(size_t s) {
    vector<double> result(actions.size());

    for (int i = 0; i < actions.size(); i++)
      result[i] = actionValue(s, actions[i]);

    return result;
  }

  void onPolicyMonteCarloControl(unsigned nIters) {
    for (unsigned iter = 0; iter < nIters; iter++) {

    }
  }

  Matrix<double> policyIncrement(size_t s, bool weightbyProb = true) {
    MatrixD result = MatrixD::zeros(actions.size(), 1);

    vector<double> actionValues = actionValuesForState(s);
    const MatrixD &currentPolicy = policyForState(s);
    double bestQ = 0;

    for (size_t i = 0; i < actionValues.size(); i++) {
      // weight action values by the probability each action is chosen
      if (weightbyProb)
        actionValues[i] *= currentPolicy(i, 0);

      // store best weighted action value
      if (i == 0)
        bestQ = actionValues[i];
      else if (actionValues[i] > bestQ)
        bestQ = actionValues[i];
    }

    // actions with best action value receive equal probability of being chosen,
    // all others have prob 0
    for (size_t i = 0; i < actions.size(); i++) {
      if (actionValues[i] == bestQ)
        result(i, 0) = 1;
    }

    return normalizeToOne(result);
  }

  string prettifyPolicy() {
    string s = "";
    for (size_t i = 0; i < value.nRows(); i++) {
      for (size_t j = 0; j < value.nCols(); ++j) {
        size_t state = fromCoord(i, j);

        if (isGoal(state)) {
          s += "☻";
          continue;
        }

        const MatrixD p = policyForState(state);
        double maxProb = p.max();

        if (p(UP, 0) == maxProb and p(DOWN, 0) == maxProb and p(LEFT, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "╬";
        else if (p(UP, 0) == maxProb and p(DOWN, 0) == maxProb and p(LEFT, 0) == maxProb)
          s += "╣";
        else if (p(UP, 0) == maxProb and p(DOWN, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "╠";
        else if (p(UP, 0) == maxProb and p(LEFT, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "╩";
        else if (p(DOWN, 0) == maxProb and p(LEFT, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "╦";
        else if (p(UP, 0) == maxProb and p(DOWN, 0) == maxProb)
          s += "║";
        else if (p(UP, 0) == maxProb and p(LEFT, 0) == maxProb)
          s += "╝";
        else if (p(UP, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "╚";
        else if (p(DOWN, 0) == maxProb and p(LEFT, 0) == maxProb)
          s += "╗";
        else if (p(DOWN, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "╔";
        else if (p(LEFT, 0) == maxProb and p(RIGHT, 0) == maxProb)
          s += "═";
        else if (p(UP, 0) == maxProb)
          s += "⇧";
        else if (p(DOWN, 0) == maxProb)
          s += "⇩";
        else if (p(LEFT, 0) == maxProb)
          s += "⇦";
        else if (p(RIGHT, 0) == maxProb)
          s += "⇨";
      }
      s += '\n';
    }
    return s;
  }

  void iterativePolicyEvaluation(double threshold, bool verbose) {

    double delta;
    do {
      delta = 0;
      for (size_t i = 0; i < value.nRows(); i++) {
        for (size_t j = 0; j < value.nCols(); j++) {
          size_t state1 = fromCoord(i, j);

          double currentV = value(i, j), newV = 0;

          for (ActionType action : actions)
            newV += policy(state1, action) * actionValue(state1, action);

          value(i, j) = newV;

          double newDelta = abs(currentV - value(i, j));

          if (newDelta > delta)
            delta = newDelta;
        }
      }
      if (verbose) cout << value << endl;
    } while (delta >= threshold);
  }

  void policyIteration(double threshold = .000001, bool verbose = true) {
    // step 1: initialization was done in the constructor

    bool stablePolicy;
    do {
      // step 2: policy evaluation
      iterativePolicyEvaluation(threshold, verbose);

      // round values to avoid precision error
      for (size_t i = 0; i < value.nRows(); ++i)
        for (size_t j = 0; j < value.nCols(); ++j)
          value(i, j) = floor(value(i, j) * 10000) / 10000;


      // step 3: policy improvement
      stablePolicy = true;
      for (size_t i = 0; i < value.nRows(); ++i) {
        for (size_t j = 0; j < value.nCols(); ++j) {
          size_t state = fromCoord(i, j);

          // retrieve policy for current state
          MatrixD currentPolicy = policyForState(state);

          // generate a better policy
          MatrixD bestPolicy = policyIncrement(state);

          if (currentPolicy != bestPolicy) {
            policy.setRow(state, bestPolicy.transpose());
            stablePolicy = false;
          }
        }
      }
      if (verbose) cout << prettifyPolicy() << endl;
    } while (!stablePolicy);
  }

  void valueIteration(double threshold = .000001, bool verbose = true) {

    double delta;
    do {
      delta = 0;
      for (size_t i = 0; i < value.nRows(); i++) {
        for (size_t j = 0; j < value.nCols(); j++) {
          size_t state1 = fromCoord(i, j);

          // workaround to optimize running time
          if (isGoal(state1))
            continue;

          vector<double> actionValues = actionValuesForState(state1);

          size_t action = 0;
          for (size_t k = 1; k < actionValues.size(); k++) {
            if (actionValues[k] > actionValues[action]) {
              action = k;
            }
          }

          double currentV = value(i, j);
          value(i, j) = actionValues[action];

          policy.setRow(state1, policyIncrement(state1, false).transpose());

          double newDelta = abs(currentV - value(i, j));

          if (newDelta > delta)
            delta = newDelta;
        }
      }
      if (verbose) cout << value << prettifyPolicy() << endl;
    } while (delta >= threshold);
  }
};

#endif //MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
