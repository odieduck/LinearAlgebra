package org.haiyang.math.java.dl.networks;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.haiyang.math.java.dl.DeltaFunctionJava;
import org.haiyang.math.java.dl.NeuralNetwork;
import org.haiyang.math.java.la.MatrixJava;
import org.haiyang.io.data.TrainingData;

import static java.util.Collections.unmodifiableList;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toUnmodifiableList;
import static org.haiyang.math.java.dl.activate.ActivateFunctionsJava.SIGMOID;
import static org.haiyang.math.java.dl.activate.ActivateFunctionsJava.SIGMOID_PRIME;

/**
 * A simple implementation of a feedforward network
 */
public class FeedForwardNeuralNetwork implements NeuralNetwork {
    private final List<Integer> networkSizes;
    private final List<MatrixJava> biases;
    private final List<MatrixJava> weights;
    private final Function<MatrixJava, MatrixJava> activate;

    /**
     * Initialize a {@link FeedForwardNeuralNetwork} with random weights and biases and a fixed network size
     *
     * @param networkSizes
     * @param activate
     */
    public FeedForwardNeuralNetwork(List<Integer> networkSizes, Function<MatrixJava, MatrixJava> activate) {
        this.activate = activate;
        // freeze the network sizes
        this.networkSizes = unmodifiableList(networkSizes);

        // build the biases vectors and feeze it
        biases = networkSizes.stream()
                .skip(1)
                .map(v -> MatrixJava.getGaussionRandomMatrix(v, 1))
                .collect(toUnmodifiableList());

        // build the weights matrixes
        List<MatrixJava> weights = new ArrayList<>();
        for (int i = 0; i < networkSizes.size() - 1; i++) {
            weights.add(MatrixJava.getGaussionRandomMatrix(networkSizes.get(i + 1), networkSizes.get(i)).mul(
                    1 / Math.sqrt(networkSizes.get(i))));
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
    public FeedForwardNeuralNetwork(List<Integer> networkSizes, List<MatrixJava> biases, List<MatrixJava> weights,
                                    Function<MatrixJava, MatrixJava> activate) {
        this.activate = activate;
        this.networkSizes = networkSizes;
        this.biases = biases;
        this.weights = weights;
    }

    @Override
    public List<Integer> getLayerSizes() {
        return this.networkSizes;
    }

    @Override
    public List<MatrixJava> getBiases() {
        return biases;
    }

    @Override
    public List<MatrixJava> getWeights() {
        return weights;
    }

    @Override
    public Function<MatrixJava, MatrixJava> getActivationFunction() {
        return activate;
    }

    /**
     * Evaluate the network against test data in {@link TrainingData} format
     * Count how many correct predictions.
     *
     * @param trainingData
     * @return
     */
    public int evaluate(List<TrainingData<MatrixJava, MatrixJava>> trainingData,
                        BiFunction<MatrixJava, MatrixJava, Boolean> evaluator) {
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
    public MatrixJava feedforward(MatrixJava input) {
        Iterator<MatrixJava> b = biases.iterator();
        Iterator<MatrixJava> w = weights.iterator();
        while (b.hasNext() && w.hasNext()) {
            input = SIGMOID.apply(w.next()
                    .dot(input)
                    .add(b.next()));
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
    private MatrixJava costDerivative(MatrixJava output, MatrixJava expected) {
        return output.minus(expected);
    }

    /**
     * A flexible way to compute delta externally
     *
     * @param costDerivative
     * @param z
     * @param delta
     * @return
     */
    private MatrixJava delta(MatrixJava costDerivative, MatrixJava z, BiFunction<MatrixJava, MatrixJava, MatrixJava> delta) {
        return delta.apply(costDerivative, z);
    }

    /**
     * Back propagation with {@Link MatrixJava} input and the {@link MatrixJava} expected output (label}
     * returns a delta {@link FeedForwardNeuralNetwork}
     *
     * @param input
     * @param expected
     * @return
     */
    public FeedForwardNeuralNetwork backprop(MatrixJava input, MatrixJava expected, DeltaFunctionJava deltaFunc) {
        List<MatrixJava> newBiases = biases.stream()
                .map(m -> new MatrixJava(m.getRowCount(), m.getColCount()))
                .collect(toList());
        List<MatrixJava> newWeights = weights.stream()
                .map(m -> new MatrixJava(m.getRowCount(), m.getColCount()))
                .collect(toList());

        // Feedforward

        // vectors of all activations layer by layer
        List<MatrixJava> activations = new ArrayList<>();
        // vectors of z vectors (value before activation) layer by layer
        List<MatrixJava> zs = new ArrayList<>();
        activations.add(input);
        MatrixJava activation = input;

        // co-iterate
        Iterator<MatrixJava> bs = getBiases().iterator();
        Iterator<MatrixJava> ws = getWeights().iterator();

        while (bs.hasNext() && ws.hasNext()) {
            MatrixJava z = ws.next()
                    .dot(activation)
                    .add(bs.next());
            activation = getActivationFunction().apply(z);
            activations.add(activation);
            zs.add(z);
        }

        // Back propagate
        MatrixJava delta = deltaFunc.delta(activations.get(activations.size() - 1), expected, zs.get(zs.size() - 1));

        newBiases.set(newBiases.size() - 1, delta);
        newWeights.set(newWeights.size() - 1, delta.dot(activations.get(activations.size() - 2)
                .transpose()));

        // backward starting from the second last layer
        // Just using indexes so it is simpler
        for (int backIndex = 2; backIndex < networkSizes.size(); backIndex++) {
            MatrixJava z = zs.get(zs.size() - backIndex);
            MatrixJava sigmoidPrime = SIGMOID_PRIME.apply(z);

            delta = weights.get(weights.size() - backIndex + 1)
                    .transpose()
                    .dot(delta)
                    .mul(sigmoidPrime);
            newBiases.set(newBiases.size() - backIndex, delta);
            newWeights.set(newWeights.size() - backIndex, delta.dot(activations.get(activations.size() - backIndex - 1)
                    .transpose()));
        }

        // generate a new network and return
        return new FeedForwardNeuralNetwork(networkSizes, newBiases, newWeights, getActivationFunction());
    }
}
