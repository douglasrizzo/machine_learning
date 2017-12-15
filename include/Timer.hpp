//
// Created by dodo on 13/12/17.
//

#ifndef MACHINE_LEARNING_TIMER_HPP
#define MACHINE_LEARNING_TIMER_HPP

#include <iostream>
#include <string>
#include <chrono>

using namespace std;

class Timer {
  using myClock = chrono::high_resolution_clock;
 private:
  unsigned int interval, predictedIters;
  float lastUpdate;
  chrono::time_point<chrono::system_clock> startTime;

 public:
  Timer(unsigned int interval=0, unsigned int predictedIters = 0) :
      interval(interval),
      predictedIters(predictedIters) {
  }

  void start() {
    this->startTime = myClock::now();
  }

  bool activate(unsigned int currentIter = 0) {
    chrono::time_point<chrono::system_clock> currentTime = myClock::now();
    float totalSeconds = ((chrono::duration<float>) (currentTime - startTime)).count();

    if (totalSeconds - lastUpdate > interval) {
      lastUpdate = totalSeconds;

      if (currentIter > 0 and predictedIters > 0) {
        float estimatedTotalSeconds = (totalSeconds / (currentIter)) * predictedIters;
        string formattedTotalTime = Timer::prettyTime(estimatedTotalSeconds - totalSeconds);

        cout << "it " << currentIter + 1 << "/" << predictedIters << " (est. " << formattedTotalTime << ")" << endl;
      }
      return true;
    }

    return false;
  }

  static string prettyTime(float secondsFloat) {
    int totalSeconds = (int) secondsFloat;
    int milliseconds = (int) (secondsFloat * 1000) % 1000;
    int seconds = totalSeconds % 60;
    int minutes = (totalSeconds / 60) % 60;
    int hours = (totalSeconds / (60 * 60)) % 24;

    std::__cxx11::string formattedTime = std::__cxx11::to_string(hours) + ":" + std::__cxx11::to_string(minutes) + ":"
        + std::__cxx11::to_string(seconds) + "." + std::__cxx11::to_string(milliseconds);
    return formattedTime;
  }

  string runningTime() {
    float totalSeconds = ((chrono::duration<float>) (myClock::now() - startTime)).count();
    return Timer::prettyTime(totalSeconds);
  }
};

#endif //MACHINE_LEARNING_TIMER_HPP
