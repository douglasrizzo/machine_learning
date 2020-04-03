//
// Created by dodo on 13/12/17.
//

#ifndef MACHINE_LEARNING_TIMER_HPP
#define MACHINE_LEARNING_TIMER_HPP

#include <iostream>
#include <string>
#include <chrono>
#include <climits>

using namespace std;


/**
 * A timer that keeps track of time, prints formatted time, checks if an interval has passed etc.
 */
class Timer {
    using myClock = chrono::high_resolution_clock;
private:
    unsigned int interval, predictedIters;
    float lastUpdate;
    chrono::time_point<chrono::system_clock> startTime;

    /**
     * Counts the number of digits in an integer
     * @param x an integer
     * @return number of digits in <code>x</code>
     */
    static int numDigits(int32_t x) {
        if(x == INT_MIN)
            return 10 + 1;

        if(x < 0)
            return numDigits(-x) + 1;

        if(x >= 10000) {
            if(x >= 10000000) {
                if(x >= 100000000) {
                    if(x >= 1000000000)
                        return 10;

                    return 9;
                }

                return 8;
            }

            if(x >= 100000) {
                if(x >= 1000000)
                    return 7;

                return 6;
            }

            return 5;
        }

        if(x >= 100) {
            if(x >= 1000)
                return 4;

            return 3;
        }

        if(x >= 10)
            return 2;

        return 1;
    }

    /**
     * Counts the number of digits in a char
     * @param n a char
     * @return number of digits in <code>n</code>
     */
    static int numDigits(char n) {
        // if you have the time, replace this with a static initialization to avoid
        // the initial overhead & unnecessary branch
        static char x[256] = {0};

        if(x[0] == 0) {
            for(char c = 1; c != 0; c++)
                x[c] = numDigits((int32_t) c);

            x[0] = 1;
        }

        return x[n];
    }

    /**
     * Pads a number with 0s on the left until it reaches a desired length
     * @param value the integer value to be zero-padded
     * @param desiredSize the desired size of the padded representation
     * @return a string with <code>value</code> zero-padded to <code>desiredSize</code> size
     */
    static string zeroPad(int value, unsigned int desiredSize) {
        return string(desiredSize - numDigits(value), '0').append(to_string(value));
    }

public:

    /**
     * Creates an instance of a timer object
     * @param interval the interval the timer will check for
     * @param predictedIters number of predicted iterations, so th etime can output a predicted time of conclusion
     */
    explicit Timer(unsigned int interval = 0, unsigned int predictedIters = 0) :
        interval(interval),
        predictedIters(predictedIters), lastUpdate(0) {}

    /**
     * Start the timer
     */
    void start() {
        this->startTime = myClock::now();
    }

    /**
     * Checks if the time interval passed in the constructor has passed
     * @param currentIter current iteration number, so an estimated conclusion time can be printed
     * @return true if the stipulated interval has passed, otherwise false
     */
    bool activate(unsigned int currentIter = 0) {
        chrono::time_point<chrono::system_clock> currentTime = myClock::now();
        float totalSeconds = ((chrono::duration<float>)(currentTime - startTime)).count();

        if(totalSeconds - lastUpdate > interval) {
            lastUpdate = totalSeconds;

            if(currentIter > 0 and predictedIters > 0) {
                float estimatedTotalSeconds = (totalSeconds / (currentIter)) * predictedIters;
                string formattedTotalTime = Timer::prettyTime(estimatedTotalSeconds - totalSeconds);

                cout << "it " << currentIter + 1 << "/" << predictedIters << " (est. " << formattedTotalTime << ")" << endl;
            }

            return true;
        }

        return false;
    }

    /**
     * Formats seconds as HH:MM:SS.mmm
     * @param secondsFloat number of seconds
     * @return a string separating the given seconds into hours, minutes, seconds and milliseconds
     */
    static string prettyTime(float secondsFloat) {
        int totalSeconds = (int) secondsFloat;
        int milliseconds = (int) (secondsFloat * 1000) % 1000;
        int seconds = totalSeconds % 60;
        int minutes = (totalSeconds / 60) % 60;
        int hours = (totalSeconds / (60 * 60)) % 24;

        string formattedTime = to_string(hours) + ":" + zeroPad(minutes, 2) + ":"
                               + zeroPad(seconds, 2) + "." + zeroPad(milliseconds, 3);
        return formattedTime;
    }

    /**
     * @return the time since the timer has been activated
     */
    string runningTime() {
        float totalSeconds = ((chrono::duration<float>)(myClock::now() - startTime)).count();
        return Timer::prettyTime(totalSeconds);
    }
};


#endif // MACHINE_LEARNING_TIMER_HPP
