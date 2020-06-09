import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import java.io.File;
import static java.lang.System.exit;

/**
 * Contains code and main method for LogisticRegression model
 */
public class LogisticRegression {
    // Class Variable
    static Dataset dataset;
    // Variables for Logistic Regression Model
    static double[] weights;
    static double bias;
    static double currentLoss = 0; // to check convergence
    static double previousLoss = 0;
    static int epoch;

    // Set constants
    final static String CSV_FILE_LOCATION = "./mnist_train.csv";
    final static String TEST_SET_LOCATION = "./test.txt";
    final static int LABEL0 = 3;
    final static int LABEL1 = 6;
    final static int FEATURE_DIMENSION = 784; // MNIST
    final static long RANDOM_SEED = 20200602;
    // Hyper-parameters
    final static double EPSILON = 0.000001;
    final static int MAX_EPOCH = 50000;
    final static double LEARNING_RATE = 0.1;

    // Logging
    static StringBuilder logger = new StringBuilder();

    /**
     * main method of LogisticRegression Class
     *
     * @param args Command Line Arguments (CLAs) 0. Question 1 saved file name
     *             (saveFeatureVector) 1. Question 2 saved file name (saveModel) 2.
     *             Question 3 saved file name (saveTestActivation) 3. Question 4
     *             saved file name (saveTestPrediction) 4. log file
     */
    public static void main(String[] args) {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));
        // Check for the number of command-line arguments
        if (args.length != 5) {
            System.out.println(
                    "Need to have five CLAs, the file names to store text file for each question and for logging.");
            exit(1);
        }

        // Create log file
        File log = new File(args[4]);
        FileWriter logWriter = null;
        try {
            if (!log.exists()) { // only when file not exists
                if (!log.createNewFile()) { // crate new file with given name
                    System.out.println("File not Created!!");
                    exit(1);
                }
            }
            logWriter = new FileWriter(log);
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }

        // create dataset and load data
        dataset = new Dataset(CSV_FILE_LOCATION, TEST_SET_LOCATION, LABEL0, LABEL1, FEATURE_DIMENSION);
        try {
            dataset.loadTrainingSet();
            dataset.loadTestingSet();
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }

        // Question 1: save Feature Vector of any one training image
        try {
            saveFeatureVector(dataset.getTrainingFeatures().get(0), args[0]);
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }

        // Initialize weights and bias randomly within [-1, 1]
        Random random = new Random(RANDOM_SEED);
        weights = new double[FEATURE_DIMENSION];
        for (int i = 0; i < FEATURE_DIMENSION; i++) {
            weights[i] = (random.nextDouble() * 2 - 1);
        }
        bias = random.nextDouble() * 2 - 1;

        // Gradient Descent
        Double[] activation = calculateActivation(dataset.getTrainingFeatures()); // Calculate activation
        for (epoch = 1; epoch <= MAX_EPOCH; epoch++) {
            // logging
            System.out.print("Epoch " + epoch + " ");
            logger.append("Epoch ").append(epoch).append(" ");

            updateWeightsAndBias(activation); // updates weight and bias
            // calculate new Activation based on the updated weights
            activation = calculateActivation(dataset.getTrainingFeatures());

            // check for convergence
            if (checkConvergence(activation)) { // converged
                System.out.println("Converged after Epoch " + epoch);
                logger.append("Converged after Epoch ").append(epoch).append("\n");
                break;
            }
        }

        // Question 2: save logistic regression weights and bias
        try {
            saveModel(args[1]);
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }

        // Testing the model
        Double[] testActivation = calculateActivation(dataset.getTestingFeatures()); // calculate activation for testing
                                                                                     // set
        // Get the prediction based on testActivation
        Integer[] testPrediction = new Integer[dataset.getTestingFeatures().size()];
        for (int dataIndex = 0; dataIndex < dataset.getTestingFeatures().size(); dataIndex++) {
            if (testActivation[dataIndex] > 0.5) { // threshold is 0.5
                testPrediction[dataIndex] = 1;
            } else {
                testPrediction[dataIndex] = 0;
            }
        }

        // Question 3: save activation values for the testing set
        try {
            saveTestActivation(args[2], testActivation);
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }

        // Question 4: save prediction result for the testing set
        try {
            saveTestPrediction(args[3], testPrediction);
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }

        // Write log file
        try {
            logWriter.write(logger.toString());
            logWriter.flush();
            logWriter.close();
        } catch (IOException e) {
            System.out.println("Interrupted I/O Operation");
            e.printStackTrace();
            exit(1);
        }
    }

    /**
     * Method to handle Question 1 - save one of the training sample as text (CSV)
     * file
     *
     * @param feature  the contents of the file - one of the training sample
     * @param filename filename of destination text (CSV) file
     * @throws IOException While writing file, IOException might be caused
     */
    private static void saveFeatureVector(Double[] feature, String filename) throws IOException {
        // create FileWriter
        File destination = new File(filename);
        if (!destination.exists()) { // only when file not exists
            if (!destination.createNewFile()) { // crate new file with given name
                System.out.println("File not Created!!");
                exit(1);
            }
        }
        FileWriter fileWriter = new FileWriter(destination);

        // create string to write
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < feature.length; i++) {
            result.append(String.format("%.2f", feature[i]));
            if (i != feature.length - 1) { // after the last number, should not add comma
                result.append(",");
            }
        }

        // Write String to file
        fileWriter.write(result.toString());
        fileWriter.flush();
        fileWriter.close();

        // Prompting message for logging
        System.out.println("Question 1: " + filename + " Saved\n");
    }

    /**
     * Private helper method to calculate activation
     *
     * @param features ArrayList<Double[]> array contains list of features (data
     *                 point)
     * @return activation values (using logistic sigmoid function)
     */
    private static Double[] calculateActivation(ArrayList<Double[]> features) {
        Double[] activation = new Double[features.size()]; // space to save all activations for epoch
        for (int dataIndex = 0; dataIndex < features.size(); dataIndex++) { // for all data entry
            // Calculate weighted sum, z
            double weightedSum = 0;
            for (int featureIndex = 0; featureIndex < FEATURE_DIMENSION; featureIndex++) {
                weightedSum += weights[featureIndex] * features.get(dataIndex)[featureIndex];
            }
            weightedSum += bias;

            // calculate activation
            activation[dataIndex] = 1.0 / (1.0 + Math.exp(-1 * weightedSum));
        }

        return activation;
    }

    /**
     * Private helper method to update Weights and Bias of logistic regression model
     *
     * @param activation previously calculated activation
     */
    private static void updateWeightsAndBias(Double[] activation) {
        // Update Weights
        for (int featureIndex = 0; featureIndex < FEATURE_DIMENSION; featureIndex++) {
            double updateWeights = 0;
            for (int dataIndex = 0; dataIndex < dataset.getTrainingFeatures().size(); dataIndex++) {
                updateWeights += (activation[dataIndex] - dataset.getTrainingLabels().get(dataIndex)) // (a_i - y_i) *
                                                                                                      // x_i
                        * dataset.getTrainingFeatures().get(dataIndex)[featureIndex];
            }
            weights[featureIndex] = weights[featureIndex] - LEARNING_RATE * updateWeights; // w - lr * updateWeight
        }

        // update Bias
        double updateBias = 0;
        for (int dataIndex = 0; dataIndex < dataset.getTrainingFeatures().size(); dataIndex++) {
            updateBias += (activation[dataIndex] - dataset.getTrainingLabels().get(dataIndex)); // a_i - y_i
        }
        bias = bias - LEARNING_RATE * updateBias; // b - lr * updateBias
    }

    /**
     * Private helper method to check convergence of the model
     *
     * @param activation activation with updated weight and bias
     * @return whether the model has been converged or not
     */
    private static boolean checkConvergence(Double[] activation) {
        previousLoss = currentLoss; // Assign previous loss

        // Calculate for currentLoss
        currentLoss = 0;
        for (int dataIndex = 0; dataIndex < dataset.getTrainingFeatures().size(); dataIndex++) {
            if (dataset.getTrainingLabels().get(dataIndex) == 1) {
                if (activation[dataIndex] < 0.0001) {
                    currentLoss += 100.0; // To prevent NaN, for the cases possibly cause Inf, just add a large number
                } else {
                    currentLoss -= Math.log(activation[dataIndex]);
                }
            } else {
                if (activation[dataIndex] > 0.9999) {
                    currentLoss += 100.0; // To prevent NaN, for the cases possibly cause Inf, just add a large number
                } else {
                    currentLoss -= Math.log(1 - activation[dataIndex]);
                }
            }
        }

        // logging
        System.out.println("Loss: " + currentLoss);
        logger.append("Loss: ").append(currentLoss).append("\n");

        // Check for convergence (Return true when abs(previousLoss - currentLoss) is
        // less than epsilon)
        return Math.abs(previousLoss - currentLoss) < EPSILON;
    }

    /**
     * Method to handle Question 2 - save model weights and bias as text (CSV) file
     *
     * @param filename filename of destination text (CSV) file
     * @throws IOException While writing file, IOException might be caused
     */
    private static void saveModel(String filename) throws IOException {
        // create FileWriter
        File destination = new File(filename);
        if (!destination.exists()) { // only when file not exists
            if (!destination.createNewFile()) { // crate new file with given name
                System.out.println("File not Created!!");
                exit(1);
            }
        }
        FileWriter fileWriter = new FileWriter(destination);

        // create string to write
        StringBuilder result = new StringBuilder();
        for (double weight : weights) { // save weights
            result.append(String.format("%.4f", weight));
            result.append(",");
        }
        result.append(String.format("%.4f", bias));

        // Write String to file
        fileWriter.write(result.toString());
        fileWriter.flush();
        fileWriter.close();

        // Prompting message for logging
        System.out.println("\nQuestion 2: " + filename + " Saved\n");
    }

    /**
     * Method to handle Question 3 - save test activation as text (CSV) file
     *
     * @param testActivation Double array containing testing set activation values
     * @param filename       filename of destination text (CSV) file
     * @throws IOException While writing file, IOException might be caused
     */
    private static void saveTestActivation(String filename, Double[] testActivation) throws IOException {
        // create FileWriter
        File destination = new File(filename);
        if (!destination.exists()) { // only when file not exists
            if (!destination.createNewFile()) { // crate new file with given name
                System.out.println("File not Created!!");
                exit(1);
            }
        }
        FileWriter fileWriter = new FileWriter(destination);

        // create string to write
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < testActivation.length; i++) {
            result.append(String.format("%.2f", testActivation[i]));
            if (i != testActivation.length - 1) { // after the last number, should not add comma
                result.append(",");
            }
        }

        // Write String to file
        fileWriter.write(result.toString());
        fileWriter.flush();
        fileWriter.close();

        // Prompting message for logging
        System.out.println("Question 3: " + filename + " Saved\n");
    }

    /**
     * Method to handle Question 4 - save test prediction result as text (CSV) file
     *
     * @param testPrediction Integer array containing testing set prediction results
     * @param filename       filename of destination text (CSV) file
     * @throws IOException While writing file, IOException might be caused
     */
    private static void saveTestPrediction(String filename, Integer[] testPrediction) throws IOException {
        // create FileWriter
        File destination = new File(filename);
        if (!destination.exists()) { // only when file not exists
            if (!destination.createNewFile()) { // crate new file with given name
                System.out.println("File not Created!!");
                exit(1);
            }
        }
        FileWriter fileWriter = new FileWriter(destination);

        // create string to write
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < testPrediction.length; i++) {
            result.append(String.format("%d", testPrediction[i]));
            if (i != testPrediction.length - 1) { // after the last number, should not add comma
                result.append(",");
            }
        }

        // Write String to file
        fileWriter.write(result.toString());
        fileWriter.flush();
        fileWriter.close();

        // Prompting message for logging
        System.out.println("Question 4: " + filename + " Saved\n");
    }

}