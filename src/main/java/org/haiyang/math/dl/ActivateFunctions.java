package org.haiyang.math.dl;

import org.haiyang.math.MathFunctions;

import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;

public final class ActivateFunctions {

    /**
     * Return derevitive of a sigmoid activation function
     *
     * @return
     */
    public static final Function<double[], double[]> SIGMOID_PRIME = input -> applyWith(input, MathFunctions.SIGMOID_PRIME);

    /**
     * Return a sigmoid activation function
     *
     * @return
     */
    public static final Function<double[], double[]> SIGMOID = input -> applyWith(input, MathFunctions.SIGMOID);

    /**
     * Given a {@link DoubleUnaryOperator}, apply to each element of input array
     *
     * @param input
     * @param func
     * @return
     */
    private static double[] applyWith(double[] input, DoubleUnaryOperator func) {
        double[] output = new double[input.length];
        for (int i = 0; i < output.length; i++) {
            output[i] = func.applyAsDouble(input[i]);
        }
        return output;
    }
}
