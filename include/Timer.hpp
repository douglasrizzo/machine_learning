//
// Created by dodo on 13/12/17.
//

#ifndef MACHINE_LEARNING_TIMER_HPP
#define MACHINE_LEARNING_TIMER_HPP

#include <iostream>
#include <string>

using namespace std;

class Timer {

 public:
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
};

#endif //MACHINE_LEARNING_TIMER_HPP
