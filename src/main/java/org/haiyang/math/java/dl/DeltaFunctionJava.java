package org.haiyang.math.java.dl;

import org.haiyang.math.dl.DeltaFunction;
import org.haiyang.math.java.dl.activate.ActivateFunctionsJava;
import org.haiyang.math.java.la.MatrixJava;

/**
 * Interface of computing delta from a, y and z
 */
public interface DeltaFunctionJava extends DeltaFunction<MatrixJava> {

    /**
     * This is a function to compute the delta value for QUADRATIC cost function
     */
    DeltaFunctionJava QUADRATIC = (a, y, z) -> a.minus(y)
            .mul(ActivateFunctionsJava.SIGMOID_PRIME.apply(y));
    /**
     * This is a function to compute the delta value for CORSS-ENTROPY cost function
     */
    DeltaFunctionJava CROSS_ENTROPY = (a, y, z) -> a.minus(y);
}
