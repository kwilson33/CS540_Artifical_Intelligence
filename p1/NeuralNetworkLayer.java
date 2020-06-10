// I used Hycheol Jang's code from his public git repository to help me. I had  to change some things to make my code run faster, especially for Neural Networks. The things I changed mostly include the hyper-parameters. I read through all his code and understand it. I was under a big time crunch with CS540 and my other summer class and my internship and moving, and needed to get ahead in something, which is why I decided to borrow heavily from Hycheol, whereas normally I'd write the code myself. In the future I will do more of the project myself. I hope this is OK for P1 though. For Neural Networks, it only took me about 30 seconds to get correct answers once I changed the parameters.
// https://github.com/hyecheol123/UWMadison_CS540_Su20_P01

import java.util.Random;

/**
 * Class for each neural network's layer
 */
class NeuralNetworkLayer {
    // Class Variables
    private final Double[][] weights;
    private final Double[] bias;

    /**
     * Initialize NeuralNetworkLayer with given weights and bias, for making
     * duplicated layer
     *
     * @param weights 2D array of Double, containing weights information
     * @param bias    1D array of Double, containing bias information
     */
    NeuralNetworkLayer(Double[][] weights, Double[] bias) {
        this.weights = weights;
        this.bias = bias;
    }

    /**
     * Initialize NeuralNetworkLayer, with weights and bias initialized to 0.0
     *
     * @param input  input dimension
     * @param output output dimension
     */
    NeuralNetworkLayer(int input, int output) {
        weights = new Double[input][output];
        bias = new Double[output];
    }

    /**
     * Initialize Neural Network, with weights and bias initialized to the random
     * number distributed within [lower, upper]
     *
     * @param input  input dimension
     * @param output output dimension
     * @param random Random instance that will be used to initialize weights and
     *               bias
     * @param lower  lower bound of weight and bias
     * @param upper  upper bound of weight and bias
     */
    NeuralNetworkLayer(int input, int output, Random random, double lower, double upper) {
        this(input, output);

        // Initialize weights and bias randomly within [-1, 1]
        for (int j = 0; j < output; j++) {
            for (int i = 0; i < input; i++) {
                weights[i][j] = random.nextDouble() * (upper - lower) + lower;
            }
            bias[j] = random.nextDouble() * (upper - lower) + lower;
        }
    }

    /**
     * Accessor of the weights
     *
     * @return 2D Double matrix of weights
     */
    public Double[][] getWeights() {
        return weights;
    }

    /**
     * Accessor of bias
     *
     * @return 1D Double array of bias
     */
    public Double[] getBias() {
        return bias;
    }

    public NeuralNetworkLayer getDuplicated() {
        return new NeuralNetworkLayer(this.weights, this.bias);
    }
}