package org.haiyang.math.util;

import java.util.Arrays;

import org.haiyang.math.dl.DeltaFunction;
import org.haiyang.math.dl.TrainingData;
import org.haiyang.math.dl.activate.ActivateFunctions;
import org.haiyang.math.dl.gd.StochasticGradientDescent;
import org.haiyang.math.dl.networks.FeedForwardNetwork;
import org.haiyang.math.la.Matrix;

/**
 * Test for a network with a simple network.
 */
public class SimpleNetwork {
    public static void main(String[] args) {
        FeedForwardNetwork feedForwardNetwork = new FeedForwardNetwork(Arrays.asList(3, 3, 1),
                ActivateFunctions.SIGMOID);

        StochasticGradientDescent sgd = new StochasticGradientDescent(Arrays.asList(
                new TrainingData<>(new Matrix(new double[][] { { 1.0 }, { 1.0 }, { 1.0 } }),
                        new Matrix(new double[][] { { 0 } }))), 300, 1, 0.3, 5.0);
        FeedForwardNetwork result = sgd.descent(feedForwardNetwork, Arrays.asList(
                new TrainingData<>(new Matrix(new double[][] { { 1.0 }, { 1.0 }, { 1.0 } }),
                        new Matrix(new double[][] { { 0 } }))), (x, y) -> {
            MatrixPerfRun.print(x);
            return Math.abs(x.get(0, 0) - y.get(0, 0)) < 1e-6;
        }, DeltaFunction.CROSS_ENTROPY);
    }
}
