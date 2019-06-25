package org.haiyang.math.java.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.haiyang.math.java.dl.activate.ActivateFunctionsJava;
import org.haiyang.math.java.dl.networks.FeedForwardNeuralNetwork;
import org.haiyang.math.java.la.MatrixJava;

/**
 * Testing utilities for FFN {@link FeedForwardNeuralNetwork}
 */
public class FFNPerfRun {

    public static void main(String[] args) {
        int N = 1000;
        FeedForwardNeuralNetwork ffn = new FeedForwardNeuralNetwork(Arrays.asList(784, 30, 10), ActivateFunctionsJava.SIGMOID);

        List<MatrixJava> input = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            input.add(MatrixJava.getGaussionRandomMatrix(784, 1));
        }

        long start = System.nanoTime();

        input.forEach(ffn::feedforward);

        System.out.println(N + " evaluation used: " + ((System.nanoTime() - start) / 1e9) + "s");
    }
}
