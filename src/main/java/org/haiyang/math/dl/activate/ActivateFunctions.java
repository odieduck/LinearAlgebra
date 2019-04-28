package org.haiyang.math.dl.activate;

import java.util.function.Function;

import org.haiyang.math.la.Matrix;
import org.haiyang.math.MathFunctions;

public final class ActivateFunctions {

    /**
     * Return derevitive of a sigmoid activation function
     *
     * @return
     */
    public static final Function<Matrix, Matrix> SIGMOID_PRIME = input -> input.transform(MathFunctions.SIGMOID_PRIME);

    /**
     * Return a sigmoid activation function
     *
     * @return
     */
    public static final Function<Matrix, Matrix> SIGMOID = input -> input.transform(MathFunctions.SIGMOID);

}
