package org.dl.java.math.java.util;

import java.util.Arrays;

import org.dl.java.math.java.dl.DeltaFunctionJava;
import org.dl.java.io.data.TrainingData;
import org.dl.java.math.java.dl.activate.ActivateFunctionsJava;
import org.dl.java.math.java.dl.gd.StochasticGradientDescent;
import org.dl.java.math.java.dl.networks.FeedForwardNeuralNetwork;
import org.dl.java.math.java.la.MatrixJava;

/**
 * Test for a network with a simple network.
 */
public class SimpleNetwork {
    public static void main(String[] args) {
        FeedForwardNeuralNetwork feedForwardNetwork = new FeedForwardNeuralNetwork(Arrays.asList(3, 3, 1),
                ActivateFunctionsJava.SIGMOID);

        StochasticGradientDescent sgd = new StochasticGradientDescent(Arrays.asList(
                new TrainingData<>(new MatrixJava(new double[][] { { 1.0 }, { 1.0 }, { 1.0 } }),
                        new MatrixJava(new double[][] { { 0 } }))), 300, 1, 0.3, 5.0);
        FeedForwardNeuralNetwork result = sgd.descent(feedForwardNetwork, Arrays.asList(
                new TrainingData<>(new MatrixJava(new double[][] { { 1.0 }, { 1.0 }, { 1.0 } }),
                        new MatrixJava(new double[][] { { 0 } }))), (x, y) -> {
            MatrixPerfRun.print(x);
            return Math.abs(x.get(0, 0) - y.get(0, 0)) < 1e-6;
        }, DeltaFunctionJava.CROSS_ENTROPY);
    }
}
