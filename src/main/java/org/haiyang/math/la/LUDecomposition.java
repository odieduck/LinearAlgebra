package org.haiyang.math.la;

/**
 * Data class representing an LU decomposition
 */
public class LUDecomposition {
    private final Matrix l;
    private final Matrix u;

    /**
     * Construct an LU
     * @param l
     * @param u
     */
    public LUDecomposition(Matrix l, Matrix u) {
        this.l = l;
        this.u = u;
    }

    /**
     * Get the L matrix
     *
     * @return
     */
    public Matrix l() {
        return l;
    }

    /**
     * Get the U matrix
     *
     * @return
     */
    public Matrix u() {
        return u;
    }
}
