package org.haiyang.math.util;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.haiyang.math.dl.activate.ActivateFunctions;
import org.haiyang.math.dl.networks.FeedForwardNetwork;
import org.haiyang.math.la.Matrix;

/**
 * Testing utilities for FFN {@link FeedForwardNetwork}
 */
public class FFNPerfRun {

    public static void main(String[] args) {
        int N = 1000;
        FeedForwardNetwork ffn = new FeedForwardNetwork(Arrays.asList(784, 30, 10), ActivateFunctions.SIGMOID);

        List<Matrix> input = new ArrayList<>();

        for (int i = 0; i < N; i++) {
            input.add(Matrix.getGaussionRandomMatrix(784, 1));
        }

        long start = System.nanoTime();

        input.forEach(ffn::feedforward);

        System.out.println(N + " evaluation used: " + ((System.nanoTime() - start) / 1e9) + "s");
    }
}
