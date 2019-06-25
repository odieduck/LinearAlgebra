package org.haiyang.math.java.dl.activate;

import java.util.function.Function;

import org.haiyang.math.java.la.MatrixJava;
import org.haiyang.math.MathFunctions;

public final class ActivateFunctionsJava {

    /**
     * Return derevitive of a sigmoid activation function
     *
     * @return
     */
    public static final Function<MatrixJava, MatrixJava> SIGMOID_PRIME = input -> input.transform(MathFunctions.SIGMOID_PRIME);

    /**
     * Return a sigmoid activation function
     *
     * @return
     */
    public static final Function<MatrixJava, MatrixJava> SIGMOID = input -> input.transform(MathFunctions.SIGMOID);

}
