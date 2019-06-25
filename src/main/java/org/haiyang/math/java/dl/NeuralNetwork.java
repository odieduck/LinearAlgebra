package org.haiyang.math.java.dl;

import java.util.List;
import java.util.function.Function;

import org.haiyang.math.java.la.MatrixJava;

/**
 * An simple interface for building a NN.
 * This API follows the example in http://neuralnetworksanddeeplearning.com/chap1.html
 */
public interface NeuralNetwork {
    /**
     * Return an {@link int[]} of layers, starting from the input layer
     *
     * @return
     */
    List<Integer> getLayerSizes();

    /**
     * Return a {@link List} of {@link MatrixJava}, an N * 1 vector
     *
     * @return
     */
    List<MatrixJava> getBiases();

    /**
     * Returns a {@link List} of {@link MatrixJava} representing the weights connecting the layers
     *
     * @return
     */
    List<MatrixJava> getWeights();

    /**
     * Get the activation function used in this network
     *
     * @return
     */
    Function<MatrixJava, MatrixJava> getActivationFunction();
}
