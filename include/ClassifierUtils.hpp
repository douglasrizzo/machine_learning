/**
 * @author Douglas De Rizzo Meneghetti (douglasrizzom@gmail.com)
 * @brief
 * @date   2017-11-23
 */

#ifndef MACHINE_LEARNING_CLASSIFIERUTILS_HPP
#define MACHINE_LEARNING_CLASSIFIERUTILS_HPP

class ClassifierUtils {
 private:
  static size_t findLabel(MatrixD y, double label) {
    for (size_t i = 0; i < y.nRows(); i++)
      if (label == y(i, 0))
        return i;

    return 0;
  }

  static MatrixD getAllClasses(const MatrixD yTrue, const MatrixD yPred) {
    MatrixD allClasses = yTrue;
    allClasses.addColumn(yPred);
    allClasses = allClasses.unique();
    allClasses.sort();
    return allClasses;
  }

 public:
  static void checkLabels(const MatrixD yTrue, const MatrixD yPred) {
    if (yTrue.nCols() != 1 or yPred.nCols() != 1)
      throw invalid_argument("Labels must be column vectors");
    if (yTrue.nRows() != yPred.nRows())
      throw invalid_argument("True labels and predicted labels must have the same size (number of rows).");
  }

  static void checkBinaryLabels(const MatrixD yTrue, const MatrixD yPred) {
    checkLabels(yTrue, yPred);
    if (!yTrue.isBinary())
      throw invalid_argument("True labels must be composed of only two classes");
    if (!yPred.isBinary())
      throw invalid_argument("Predicted labels must be composed of only two classes");
  }

  static MatrixI binarize(MatrixD m, double trueLabel) {
    return m == trueLabel;
  }

  static MatrixI confusionMatrix(MatrixD yTrue, MatrixD yPred) {
    checkLabels(yTrue, yPred);

    MatrixD allClasses = getAllClasses(yTrue, yPred);

    MatrixI result = MatrixI::zeros(allClasses.nRows(), allClasses.nRows());

    for (size_t i = 0; i < yTrue.nRows(); i++) {
      size_t tLabel = findLabel(allClasses, yTrue(i, 0));
      size_t pLabel = findLabel(allClasses, yPred(i, 0));

      result(pLabel, tLabel) += 1;
    }

    return result;
  }

  static double accuracy(MatrixD yTrue, MatrixD yPred) {
    checkLabels(yTrue, yPred);
    double accuracy = 0;
    for (size_t i = 0; i < yTrue.nRows(); i++)
      accuracy += yTrue(i, 0) == yPred(i, 0);

    return accuracy / yTrue.nRows();
  }

  static double precision(MatrixD yTrue, MatrixD yPred) {
    checkBinaryLabels(yTrue, yPred);
    MatrixI cm = confusionMatrix(yTrue, yPred);
    return cm(1, 1) / ((double) cm(1, 1) + cm(1, 0));
  }

  static double recall(MatrixD yTrue, MatrixD yPred) {
    checkBinaryLabels(yTrue, yPred);
    MatrixI cm = confusionMatrix(yTrue, yPred);
    return cm(1, 1) / ((double) cm(1, 1) + cm(0, 1));
  }

  static double f_score(MatrixD yTrue, MatrixD yPred) {
    checkBinaryLabels(yTrue, yPred);
    double p = precision(yTrue, yPred), r = recall(yTrue, yPred);
    return 2 * ((p * r) / (p + r));
  }
};

#endif //MACHINE_LEARNING_CLASSIFIERUTILS_HPP
