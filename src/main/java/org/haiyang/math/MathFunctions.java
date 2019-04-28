package org.haiyang.math;

import java.util.function.DoubleUnaryOperator;

import static java.lang.Math.exp;

/**
 * Define Math function
 */
public class MathFunctions {

    /**
     * Sigmoid function
     */
    public static final DoubleUnaryOperator SIGMOID = v -> 1.0 / (1.0 + exp(-v));

    /**
     * Derivative of the sigmoid function
     */
    public static final DoubleUnaryOperator SIGMOID_PRIME = v -> {
        double tmp = SIGMOID.applyAsDouble(v);
        return tmp * (1 - tmp);
    };
}
