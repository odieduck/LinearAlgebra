package org.haiyang.math.dl;

import org.haiyang.io.data.TrainingData;
import org.haiyang.math.la.MatrixJNI;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import static java.util.Collections.unmodifiableList;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toUnmodifiableList;
import static org.haiyang.math.dl.ActivateFunctions.SIGMOID;

/**
 * A FFN implemented with netlib JNI
 */
public class FeedForwardNetwork {
    private final List<Integer> networkSizes;
    private final List<double[]> biases;
    private final List<double[]> weights;
    private final Function<double[], double[]> activate;

    public FeedForwardNetwork(List<Integer> networkSizes, Function<double[], double[]> activate) {
        this.networkSizes = networkSizes;
        this.activate = activate;

        // build the biases vectors and freeze it
        biases = networkSizes.stream()
                .skip(1)
                .map(v -> MatrixJNI.getGaussionRandomMatrix(v, 1, 0, 1))
                .collect(toUnmodifiableList());

        // build the weights matrixes
        List<double[]> weights = new ArrayList<>();
        for (int i = 0; i < networkSizes.size() - 1; i++) {
            double[] arr = MatrixJNI.getGaussionRandomMatrix(networkSizes.get(i + 1), networkSizes.get(i), 0, 1);
            // regularization
            double regValue = 1 / Math.sqrt(networkSizes.get(i));
            double[] res = MatrixJNI.mul(arr, regValue);
            weights.add(res);
        }

        // freeze the weights
        this.weights = unmodifiableList(weights);
    }

    /**
     * Creating a new network with existing netowrk values.
     *
     * @param networkSizes
     * @param biases
     * @param weights
     * @param activate
     */
    public FeedForwardNetwork(List<Integer> networkSizes, List<double[]> biases, List<double[]> weights,
                              Function<double[], double[]> activate) {
        this.activate = activate;
        this.networkSizes = networkSizes;
        this.biases = biases;
        this.weights = weights;
    }

    public List<Integer> getLayerSizes() {
        return this.networkSizes;
    }

    public List<double[]> getBiases() {
        return biases;
    }

    public List<double[]> getWeights() {
        return weights;
    }

    public Function<double[], double[]> getActivationFunction() {
        return activate;
    }

    /**
     * Evaluate the network against test data in {@link TrainingData} format
     * Count how many correct predictions.
     *
     * @param trainingData
     * @return
     */
    public int evaluate(List<TrainingData<double[], double[]>> trainingData,
                        BiFunction<double[], double[], Boolean> evaluator) {
        return trainingData.stream()
                .map(t -> evaluator.apply(feedforward(t.getX()), t.getY()) ? 1 : 0)
                .reduce((a, b) -> a + b)
                .orElse(0);
    }

    /**
     * Given an input vector, compute the output of the current matrix
     *
     * @param input
     * @return
     */
    public double[] feedforward(double[] input) {
        // loop using index
        for (int i = 0; i < biases.size(); i++) {
            double[] bias = biases.get(i);
            double[] weight = weights.get(i);
            double[] tmp = MatrixJNI.dgemv(1.0, weight, networkSizes.get(i + 1), networkSizes.get(i), input, 1.0, bias);
            input = SIGMOID.apply(tmp);
        }

        return input;
    }

    /**
     * Computes the partial derivative of the output of activations.
     *
     * @param output   (output of the last layer activation)
     * @param expected (y)
     * @return
     */
    private double[] costDerivative(double[] output, double[] expected) {
        return MatrixJNI.minus(output, expected);
    }

    /**
     * A flexible way to compute delta externally
     *
     * @param costDerivative
     * @param z
     * @param delta
     * @return
     */
    private double[] delta(double[] costDerivative, double[] z, BiFunction<double[], double[], double[]> delta) {
        return delta.apply(costDerivative, z);
    }

    /**
     * Back propagation with {@Link double[]} input and the {@link double[]} expected output (label}
     * returns a delta {@link FeedForwardNetwork}
     *
     * @param input
     * @param expected
     * @return
     */
    public FeedForwardNetwork backprop(double[] input, double[] expected, DeltaFunction<double[]> deltaFunc) {
        List<double[]> newBiases = biases.stream()
                .map(m -> new double[m.length])
                .collect(toList());
        List<double[]> newWeights = weights.stream()
                .map(m -> new double[m.length])
                .collect(toList());

        // Feedforward

        // vectors of all activations layer by layer
        List<double[]> activations = new ArrayList<>();
        // vectors of z vectors (value before activation) layer by layer
        List<double[]> zs = new ArrayList<>();
        activations.add(input);
        double[] activation = input;

        // use index to co-iterate
        for (int i = 0; i < biases.size(); i++) {
            double[] bias = biases.get(i);
            double[] weight = weights.get(i);
            double[] z = MatrixJNI.dgemv(1.0, weight, networkSizes.get(i + 1), networkSizes.get(i), activation, 1.0, bias);
            activation = getActivationFunction().apply(z);
            activations.add(activation);
            zs.add(z);
        }

        // end feed forward

        // Back propagate
        double[] delta = deltaFunc.delta(activations.get(activations.size() - 1), expected, zs.get(zs.size() - 1));

        newBiases.set(newBiases.size() - 1, delta);
        // compute weight delta
        double[] weightDelta = MatrixJNI.daxpy(1.0, delta, activations.get(activations.size() - 2));
        newWeights.set(newWeights.size() - 1, weightDelta);

        // backward starting from the second last layer
        // Just using indexes so it is simpler
        for (int backIndex = 2; backIndex < networkSizes.size(); backIndex++) {
            double[] z = zs.get(zs.size() - backIndex);
            double[] sigmoidPrime = ActivateFunctions.SIGMOID_PRIME.apply(z);

            delta = MatrixJNI.mul(1.0, sigmoidPrime, MatrixJNI.daxpy(1.0, weights.get(weights.size() - backIndex + 1), delta));

            newBiases.set(newBiases.size() - backIndex, delta);
            newWeights.set(newWeights.size() - backIndex, MatrixJNI.daxpy(1.0, delta, activations.get(activations.size() - backIndex - 1)));
        }

        // generate a new network and return
        return new FeedForwardNetwork(networkSizes, newBiases, newWeights, getActivationFunction());
    }
}
