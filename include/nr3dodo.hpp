//
// Created by dodo on 24/10/17.
//

#ifndef MACHINE_LEARNING_NR3DODO_HPP
#define MACHINE_LEARNING_NR3DODO_HPP

//! \return a value with the same magnitude as a and the same sign as b
template<class T>
inline const T sign(T a, T b) {
  double s = (b > 0) - (b < 0);
  return a * s;
}

//! \return a squared a
template<class T>
inline const T SQR(T a) {
  return a * a;
}

#endif //MACHINE_LEARNING_NR3DODO_HPP
