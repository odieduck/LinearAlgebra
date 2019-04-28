package org.haiyang.math.dl;

import org.haiyang.math.dl.activate.ActivateFunctions;
import org.haiyang.math.la.Matrix;

/**
 * Interface of computing delta from a, y and z
 */
public interface DeltaFunction {

    /**
     * This is a function to compute the delta value for QUADRATIC cost function
     */
    DeltaFunction QUADRATIC = (a, y, z) -> a.minus(y)
                                            .mul(ActivateFunctions.SIGMOID_PRIME.apply(y));
    /**
     * This is a function to compute the delta value for CORSS-ENTROPY cost function
     */
    DeltaFunction CROSS_ENTROPY = (a, y, z) -> a.minus(y);

    /**
     * Compute the delta value given output, expected and z vector
     *
     * @param a
     * @param y
     * @param z
     * @return
     */
    Matrix delta(Matrix a, Matrix y, Matrix z);
}
