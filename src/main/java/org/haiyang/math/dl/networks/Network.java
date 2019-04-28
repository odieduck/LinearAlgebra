package org.haiyang.math.dl.networks;

import java.util.List;
import java.util.function.Function;

import org.haiyang.math.la.Matrix;

/**
 * An simple interface for building a NN.
 * This API follows the example in http://neuralnetworksanddeeplearning.com/chap1.html
 */
public interface Network {
    /**
     * Return an {@link int[]} of layers, starting from the input layer
     *
     * @return
     */
    List<Integer> getLayerSizes();

    /**
     * Return a {@link List} of {@link Matrix}, an N * 1 vector
     *
     * @return
     */
    List<Matrix> getBiases();

    /**
     * Returns a {@link List} of {@link Matrix} representing the weights connecting the layers
     *
     * @return
     */
    List<Matrix> getWeights();

    /**
     * Get the activation function used in this network
     *
     * @return
     */
    Function<Matrix, Matrix> getActivationFunction();
}
