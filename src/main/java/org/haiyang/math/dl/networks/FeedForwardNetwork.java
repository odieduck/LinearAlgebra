package org.haiyang.math.dl.networks;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;

import org.haiyang.math.la.Matrix;
import org.haiyang.math.dl.DeltaFunction;
import org.haiyang.math.dl.TrainingData;

import static java.util.Collections.unmodifiableList;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toUnmodifiableList;
import static org.haiyang.math.dl.activate.ActivateFunctions.SIGMOID;
import static org.haiyang.math.dl.activate.ActivateFunctions.SIGMOID_PRIME;
import static org.haiyang.math.la.Matrix.getGaussionRandomMatrix;

/**
 * A simple implementation of a feedforward network
 */
public class FeedForwardNetwork implements Network {
    private final List<Integer> networkSizes;
    private final List<Matrix> biases;
    private final List<Matrix> weights;
    private final Function<Matrix, Matrix> activate;

    /**
     * Initialize a {@link FeedForwardNetwork} with random weights and biases and a fixed network size
     *
     * @param networkSizes
     * @param activate
     */
    public FeedForwardNetwork(List<Integer> networkSizes, Function<Matrix, Matrix> activate) {
        this.activate = activate;
        // freeze the network sizes
        this.networkSizes = unmodifiableList(networkSizes);

        // build the biases vectors and feeze it
        biases = networkSizes.stream()
                             .skip(1)
                             .map(v -> Matrix.getGaussionRandomMatrix(v, 1))
                             .collect(toUnmodifiableList());

        // build the weights matrixes
        List<Matrix> weights = new ArrayList<>();
        for (int i = 0; i < networkSizes.size() - 1; i++) {
            weights.add(Matrix.getGaussionRandomMatrix(networkSizes.get(i + 1), networkSizes.get(i)).mul(
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
    public FeedForwardNetwork(List<Integer> networkSizes, List<Matrix> biases, List<Matrix> weights,
            Function<Matrix, Matrix> activate) {
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
    public List<Matrix> getBiases() {
        return biases;
    }

    @Override
    public List<Matrix> getWeights() {
        return weights;
    }

    @Override
    public Function<Matrix, Matrix> getActivationFunction() {
        return activate;
    }

    /**
     * Evaluate the network against test data in {@link TrainingData} format
     * Count how many correct predictions.
     *
     * @param trainingData
     * @return
     */
    public int evaluate(List<TrainingData<Matrix, Matrix>> trainingData,
            BiFunction<Matrix, Matrix, Boolean> evaluator) {
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
    public Matrix feedforward(Matrix input) {
        Iterator<Matrix> b = biases.iterator();
        Iterator<Matrix> w = weights.iterator();
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
    private Matrix costDerivative(Matrix output, Matrix expected) {
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
    private Matrix delta(Matrix costDerivative, Matrix z, BiFunction<Matrix, Matrix, Matrix> delta) {
        return delta.apply(costDerivative, z);
    }

    /**
     * Back propagation with {@Link Matrix} input and the {@link Matrix} expected output (label}
     * returns a delta {@link FeedForwardNetwork}
     *
     * @param input
     * @param expected
     * @return
     */
    public FeedForwardNetwork backprop(Matrix input, Matrix expected, DeltaFunction deltaFunc) {
        List<Matrix> newBiases = biases.stream()
                                       .map(m -> new Matrix(m.getRowCount(), m.getColCount()))
                                       .collect(toList());
        List<Matrix> newWeights = weights.stream()
                                         .map(m -> new Matrix(m.getRowCount(), m.getColCount()))
                                         .collect(toList());

        // Feedforward

        // vectors of all activations layer by layer
        List<Matrix> activations = new ArrayList<>();
        // vectors of z vectors (value before activation) layer by layer
        List<Matrix> zs = new ArrayList<>();
        activations.add(input);
        Matrix activation = input;

        // co-iterate
        Iterator<Matrix> bs = getBiases().iterator();
        Iterator<Matrix> ws = getWeights().iterator();

        while (bs.hasNext() && ws.hasNext()) {
            Matrix z = ws.next()
                         .dot(activation)
                         .add(bs.next());
            activation = getActivationFunction().apply(z);
            activations.add(activation);
            zs.add(z);
        }

        // Back propagate
        Matrix delta = deltaFunc.delta(activations.get(activations.size() - 1), expected, zs.get(zs.size() - 1));

        newBiases.set(newBiases.size() - 1, delta);
        newWeights.set(newWeights.size() - 1, delta.dot(activations.get(activations.size() - 2)
                                                                   .transpose()));

        // backward starting from the second last layer
        // Just using indexes so it is simpler
        for (int backIndex = 2; backIndex < networkSizes.size(); backIndex++) {
            Matrix z = zs.get(zs.size() - backIndex);
            Matrix sigmoidPrime = SIGMOID_PRIME.apply(z);

            delta = weights.get(weights.size() - backIndex + 1)
                           .transpose()
                           .dot(delta)
                           .mul(sigmoidPrime);
            newBiases.set(newBiases.size() - backIndex, delta);
            newWeights.set(newWeights.size() - backIndex, delta.dot(activations.get(activations.size() - backIndex - 1)
                                                                               .transpose()));
        }

        // generate a new network and return
        return new FeedForwardNetwork(networkSizes, newBiases, newWeights, getActivationFunction());
    }
}
