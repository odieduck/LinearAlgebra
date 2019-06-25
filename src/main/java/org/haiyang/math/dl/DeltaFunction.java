package org.haiyang.math.dl;

/**
 * A delta function for matrix type {@link M}
 *
 * @param <M>
 */
public interface DeltaFunction<M> {
    /**
     * Compute the delta value given output, expected and z vector
     *
     * @param a
     * @param y
     * @param z
     * @return
     */
    M delta(M a, M y, M z);
}
