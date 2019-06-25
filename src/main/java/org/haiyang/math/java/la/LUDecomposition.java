package org.haiyang.math.java.la;

/**
 * Data class representing an LU decomposition
 */
public class LUDecomposition {
    private final MatrixJava l;
    private final MatrixJava u;

    /**
     * Construct an LU
     * @param l
     * @param u
     */
    public LUDecomposition(MatrixJava l, MatrixJava u) {
        this.l = l;
        this.u = u;
    }

    /**
     * Get the L matrix
     *
     * @return
     */
    public MatrixJava l() {
        return l;
    }

    /**
     * Get the U matrix
     *
     * @return
     */
    public MatrixJava u() {
        return u;
    }
}
