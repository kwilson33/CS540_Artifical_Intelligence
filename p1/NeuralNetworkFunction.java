// I used Hycheol Jang's code from his public git repository to help me. I had  to change some things to make my code run faster, especially for Neural Networks. The things I changed mostly include the hyper-parameters. I read through all his code and understand it. I was under a big time crunch with CS540 and my other summer class and my internship and moving, and needed to get ahead in something, which is why I decided to borrow heavily from Hycheol, whereas normally I'd write the code myself. In the future I will do more of the project myself. I hope this is OK for P1 though. For Neural Networks, it only took me about 30 seconds to get correct answers once I changed the parameters.
// https://github.com/hyecheol123/UWMadison_CS540_Su20_P01

/**
 * Private inner class having static functions that will be used for gradient
 * descent
 */
class NeuralNetworkFunction {
    /**
     * Calculate logistic sigmoid function
     * 
     * @param weightedSum input for the function
     * @return return of logistic sigmoid function
     */
    public static double logisticSigmoid(double weightedSum) {
        return 1.0 / (1.0 + Math.exp(-1.0 * weightedSum));
    }

    /**
     * Differentiation of Logistic Sigmoid, g'(x) = g(x) * (1 - g(x))
     *
     * @param activation output of logistic sigmoid function (previously calculated)
     * @return differentiation of logistic sigmoid
     */
    public static double diffLogisticSigmoid(double activation) {
        return activation * (1.0 - activation);
    }

    /**
     * calculate binary cross entropy for each data point (prediction) C = -{y *
     * log(a) + (1 - y) * log(1 - a)}
     *
     * @param prediction predicted output (activation)
     * @param label      truth label
     * @return calculation of binary cross entropy
     */
    public static double binaryCrossEntropy(double prediction, int label) {
        if (label == 1) {
            if (prediction < 0.0001) { // To prevent NaN and Inf, for the cases possibly cause Inf, just add a large
                                       // number
                return 100;
            } else {
                return (-1.0) * Math.log(prediction);
            }
        } else { // when label is 0
            if (prediction > 0.9999) { // To prevent NaN and Inf, for the cases possibly cause Inf, just add a large
                                       // number
                return 100;
            } else {
                return (-1.0) * Math.log(1.0 - prediction);
            }
        }
    }
}