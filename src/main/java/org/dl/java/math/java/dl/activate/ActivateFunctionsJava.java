package org.dl.java.math.java.dl.activate;

import java.util.function.Function;

import org.dl.java.math.java.la.MatrixJava;
import org.dl.java.math.MathFunctions;

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
