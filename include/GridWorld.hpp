/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief  Class to solve the grid world toy problem using dynamic programming
 * @date   2017-12-04
 */


#ifndef MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
#define MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP

#include "Matrix.hpp"
#include "MersenneTwister.hpp"
#include "Timer.hpp"

class GridWorld {
 private:
  MatrixD V, Q, rewards, policy;
  double gamma;
  unsigned long nStates;
  vector<pair<size_t, size_t>> goals;

  enum ActionType { UP = 0, DOWN = 1, LEFT = 2, RIGHT = 3 };
  vector<ActionType> actions = {UP, DOWN, LEFT, RIGHT};

  void initialize(size_t height, size_t width, vector<pair<size_t, size_t>> goals, double gamma = 1) {
    if (goals.size() == 0)
      throw invalid_argument("No goal state, must pass at least one");

    this->nStates = height * width;

    // state and actions value functions starts as 0 everywhere
    V = MatrixD::zeros(height, width);
    Q = MatrixD::zeros(nStates, actions.size());

    // set goal rewards are 0, all other states are -1
    rewards = MatrixD::fill(height, width, -1);
    for (auto goal:goals)
      rewards(goal.first, goal.second) = 0;

    if (rewards.unique().sum() == 0)
      throw invalid_argument("All states are goal!");

    this->goals = goals;
    this->gamma = gamma;

    // initialize the policy matrix giving equal probability of choice for every action
    policy = MatrixD::fill(height * width, actions.size(), 1.0 / actions.size());
  }

  /**
   * Transforms row x column coordinates from the grid world into a raster representation
   * @param row
   * @param col
   * @return
   */
  size_t fromCoord(size_t row, size_t col) {
    return row * V.nCols() + col;
  }

  /**
   * Transforms a raster coordinate from the grid world into its corresponding row x column representation
   * @param s
   * @return
   */
  pair<size_t, size_t> toCoord(size_t s) {
    size_t row = static_cast<size_t>(ceil((s + 1) / (double) V.nCols()) - 1);
    size_t col = s - row * V.nCols();

    return {row, col};
  }

  double actionValue(size_t s, ActionType a) {
    double q = 0;

    pair<size_t, size_t> coords = toCoord(s);
    double r = rewards(coords.first, coords.second);
    for (size_t i = 0; i < V.nRows(); ++i) {
      for (size_t j = 0; j < V.nCols(); ++j) {
        size_t s2 = fromCoord(i, j);
        double t = transition(s, a, s2);
        double v = V(i, j);

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

  Matrix<double> policyIncrement(size_t s) {
    MatrixD result = MatrixD::zeros(actions.size(), 1);

    vector<double> actionValues = actionValuesForState(s);
    const MatrixD &currentPolicy = policyForState(s);
    double bestQ = actionValues[0];

    for (size_t i = 1; i < actionValues.size(); i++) {
      // store best weighted action value
      if (actionValues[i] > bestQ)
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
    for (size_t i = 0; i < V.nRows(); i++) {
      for (size_t j = 0; j < V.nCols(); ++j) {
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
    int iter = 0;
    do {
      iter++;
      delta = 0;
      for (size_t i = 0; i < V.nRows(); i++) {
        for (size_t j = 0; j < V.nCols(); j++) {
          size_t state1 = fromCoord(i, j);

          double currentV = V(i, j), newV = 0;

          for (ActionType action : actions)
            newV += policy(state1, action) * actionValue(state1, action);

          V(i, j) = newV;

          double newDelta = abs(currentV - V(i, j));

          if (newDelta > delta)
            delta = newDelta;
        }
      }
    } while (delta >= threshold);
    if (verbose)
      cout << iter << " iterations of policy evaluation" << endl;
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
        s2row = s1row != V.nCols() - 1 ? s1row + 1 : s1row;
        break;
      case LEFT:s2row = s1row;
        s2col = s1col != 0 ? s1col - 1 : s1col;
        break;
      case RIGHT:s2row = s1row;
        s2col = s1col != V.nRows() - 1 ? s1col + 1 : s1col;
        break;
      default:return 0;
    }

    return fromCoord(s2row, s2col);
  }

  MatrixD policyForState(size_t s) {
    return normalizeToOne(policy.getRow(s));
  }

  size_t getNonGoalState() {
    MersenneTwister t;
    size_t s;

    do {
      s = static_cast<size_t>(t.i_random(static_cast<int>(nStates - 1)));
    } while (isGoal(s));

    return s;
  }

  ActionType eGreedy(size_t s, double epsilon) {
    MersenneTwister t;

    if (t.d_random() <= epsilon) {
      ActionType bestAction = UP;
      double bestQ = Q(s, 0);

      for (size_t j = 1; j < actions.size(); j++) {
        if (bestQ < Q(s, j)) {
          bestQ = Q(s, j);
          bestAction = actions[j];
        }
      }

      return bestAction;

    } else {
      return actions[t.i_random(static_cast<int>(actions.size() - 1))];
    }
  }

  double bestQForState(size_t s) {
    double bestQ = Q(s, 0);

    for (size_t j = 1; j < actions.size(); j++) {
      if (bestQ < Q(s, j)) {
        bestQ = Q(s, j);
      }
    }
    return bestQ;
  }

  void getOptimalPolicyFromQ() {
    for (size_t state = 0; state < nStates; state++) {
      // store best action value for the current state
      double bestQ = Q(state, 0);
      for (size_t j = 1; j < actions.size(); j++) {
        if (bestQ < Q(state, j))
          bestQ = Q(state, j);
      }

      // actions with best action value receive equal probability of being chosen,
      // all others have prob 0
      for (size_t j = 0; j < actions.size(); j++) {
        if (Q(state, j) == bestQ)
          policy(state, j) = 1;
      }

      policy.setRow(state, normalizeToOne(policy.getRow(state).transpose()));
    }
  }

 public:

  void policyIteration(size_t height,
                       size_t width,
                       vector<pair<size_t, size_t>> goals,
                       double gamma = 1,
                       double threshold = .001,
                       bool verbose = true) {
    // step 1: initialization
    initialize(height, width, goals, gamma);

    bool stablePolicy;
    int iter = 0;
    MatrixD currentPolicy;

    do {
      currentPolicy = policy;
      iter++;
      // step 2: policy evaluation
      iterativePolicyEvaluation(threshold, verbose);


      // step 3: policy improvement
      for (size_t i = 0; i < V.nRows(); ++i) {
        for (size_t j = 0; j < V.nCols(); ++j) {
          size_t state = fromCoord(i, j);

          // workaround to optimize running time
          if (isGoal(state))
            continue;

          // retrieve policy for current state
          MatrixD currentStatePolicy = policyForState(state);

          // generate a better policy
          MatrixD betterStatePolicy = policyIncrement(state);

          if (currentStatePolicy != betterStatePolicy)
            policy.setRow(state, betterStatePolicy.transpose());
        }
      }
      if (verbose) cout << "iteration " << iter << " of policy improvement" << endl << prettifyPolicy() << endl;
    } while (currentPolicy != policy);
  }

  void valueIteration(size_t height,
                      size_t width,
                      vector<pair<size_t, size_t>> goals,
                      double gamma = 1,
                      double threshold = .000001,
                      bool verbose = true) {
    initialize(height, width, goals, gamma);
    double delta;
    int iter = 0;

    do {
      iter++;
      delta = 0;
      for (size_t i = 0; i < V.nRows(); i++) {
        for (size_t j = 0; j < V.nCols(); j++) {
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

          double currentV = V(i, j);
          V(i, j) = actionValues[action];

          policy.setRow(state1, policyIncrement(state1).transpose());

          double newDelta = abs(currentV - V(i, j));

          if (newDelta > delta)
            delta = newDelta;
        }
      }
      if (verbose) cout << "iteration " << iter << endl << prettifyPolicy() << endl;
    } while (delta >= threshold);
  }

  void MonteCarloEstimatingStarts(size_t height,
                                  size_t width,
                                  vector<pair<size_t, size_t>> goals,
                                  double gamma = 1,
                                  unsigned maxIters = 10000,
                                  bool verbose = true) {
    initialize(height, width, goals, gamma);

    MatrixI visits = MatrixI::zeros(nStates, actions.size());
    MatrixD QSum = MatrixD::zeros(nStates, actions.size());

    MersenneTwister twister;
    ActionType action;
    size_t state;
    chrono::time_point<chrono::system_clock> start = myClock::now();
    Timer timer(1, maxIters);
    timer.start();

    for (unsigned iter = 0; iter < maxIters; iter++) {

      vector<size_t> visitedStates;
      vector<ActionType> appliedActions;

      // select a random initial state
      state = getNonGoalState();

      // loop that generates the current episode
      do {
        // select random action
        action = actions[twister.i_random(static_cast<int>(actions.size() - 1))];

        // store the current state and action
        visitedStates.push_back(state);
        appliedActions.push_back(action);

        // generate next state
        state = applyAction(state, action);
      } while (!isGoal(state)); // terminate when a goal state is generated

      vector<size_t> processedStates;
      vector<ActionType> processedActions;

      double G = 0;

      // backwards loop that will update Q values
      for (size_t i = visitedStates.size() - 1; i != (size_t) 0; i--) {
        state = visitedStates[i];
        action = appliedActions[i];

        // skip this state/action pair if it has already appeared in the episode
        bool processed = false;
        for (int j = 0; j < processedStates.size(); j++) {
          if (processedStates[i] == state and processedActions[i] == action) {
            processed = true;
            break;
          }
        }

        if (processed)
          continue;

        processedStates.push_back(state);
        processedActions.push_back(action);

        pair<size_t, size_t> stateCoords = toCoord(state);

        G += rewards(stateCoords.first, stateCoords.second);

        QSum(state, action) += G;
        visits(state, action)++;
      }

      if (iter == maxIters - 1 or (verbose and timer.activate(iter))) {
        for (state = 0; state < nStates; state++) {
          // build Q matrix from sums and n. visits
          for (size_t j = 0; j < actions.size(); j++) {
            Q(state, j) = isGoal(state) ? 0 : QSum(state, j) / visits(state, j);
          }
        }

        getOptimalPolicyFromQ();

        if (verbose)
          cout << prettifyPolicy() << endl;
      }
    }
  }

  void Sarsa(size_t height, size_t width,
             vector<pair<size_t, size_t>> goals,
             double gamma = 1,
             double alpha = .3,
             double epsilon = .8,
             unsigned int maxIters = 10000,
             bool verbose = true) {
    initialize(height, width, goals, gamma);
    Timer timer(1, maxIters);
    timer.start();

    for (unsigned int iter = 0; iter < maxIters; iter++) {
      size_t s = getNonGoalState();
      ActionType a = eGreedy(s, epsilon);

      while (!isGoal(s)) {
        pair<size_t, size_t> stateCoords = toCoord(s);
        double r = rewards(stateCoords.first, stateCoords.second);
        size_t newState = applyAction(s, a);
        ActionType newAction = eGreedy(newState, epsilon);
        Q(s, a) += alpha * (r + gamma * Q(newState, newAction) - Q(s, a));
        s = newState;
        a = newAction;
      }

      if (iter == maxIters - 1 or (verbose and timer.activate(iter))) {
        getOptimalPolicyFromQ();
        if (verbose)
          cout << prettifyPolicy();
      }
    }
  }

  void QLearning(size_t height, size_t width,
                 vector<pair<size_t, size_t>> goals,
                 double gamma = 1,
                 double alpha = .3,
                 double epsilon = .8,
                 unsigned int maxIters = 10000,
                 bool verbose = true) {
    initialize(height, width, goals, gamma);
    Timer timer(1, maxIters);
    timer.start();

    for (unsigned int iter = 0; iter < maxIters; iter++) {
      size_t s = getNonGoalState();

      while (!isGoal(s)) {
        pair<size_t, size_t> stateCoords = toCoord(s);

        ActionType a = eGreedy(s, epsilon);
        double r = rewards(stateCoords.first, stateCoords.second);

        size_t newState = applyAction(s, a);

        Q(s, a) += alpha * (r + gamma * bestQForState(newState) - Q(s, a));
        s = newState;
      }

      if (iter == maxIters - 1 or (verbose and timer.activate(iter))) {
        getOptimalPolicyFromQ();
        if (verbose)
          cout << prettifyPolicy();
      }
    }
  }
};

#endif //MACHINE_LEARNING_DYNAMICPROGRAMMING_HPP
