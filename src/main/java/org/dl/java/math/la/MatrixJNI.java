package org.dl.java.math.la;

import com.github.fommil.netlib.BLAS;

import java.util.Random;

/**
 * A matrix implementation using netlib JNI
 */
public final class MatrixJNI {
    /**
     * Get the implementation of BLAS
     */
    private static final BLAS blas = BLAS.getInstance();

    /**
     * Factory method to return a Gaussian random values matrix. The gaussian distribution is
     * with mean of mean and variance of std^2.
     *
     * @param row
     * @param col
     * @param mean
     * @param std
     * @return
     */
    public static double[] getGaussionRandomMatrix(int row, int col, double mean, double std) {
        Random random = new Random();
        double[] data = new double[row * col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[j + i * col] = random.nextGaussian() * std + mean;
            }
        }

        return data;
    }

    /**
     * Produce an n*n identify matrix with
     *
     * @param n
     * @return
     */
    public static double[] identity(int n) {
        double[] data = new double[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    data[j + i * n] = 1;
                }
            }
        }
        return data;
    }

    /**
     * Wraps @{link BLAS#daxpy} y = ax + y;
     * a = val, y = [0...0]
     *
     * @param in
     * @param val
     * @return
     */
    public static double[] mul(double[] in, double val) {
        double[] out = new double[in.length];
        blas.daxpy(in.length, val, in, 0, out, 0);
        return out;
    }

    /**
     * y = a*x*y.
     * a is a scalar.
     * x, y are vectors and we do hadamard product (element wise product)
     *
     * @param left
     * @param right
     * @return
     */
    public static double[] mul(double val, double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int i = 0; i < left.length; i++) {
            out[i] = val * left[i] * right[i];
        }

        return out;
    }

    /**
     * x + y
     * x is a vector
     * y is a vector
     *
     * @param left
     * @param right
     * @return
     */
    public static double[] add(double[] left, double[] right) {
        return daxpy(1.0, left, right);
    }

    /**
     * x - y
     * x is a vector
     * y is a vector
     *
     * @param left
     * @param right
     * @return
     */
    public static double[] minus(double[] left, double[] right) {
        return daxpy(-1.0, right, left);
    }

    /**
     * Performs alpha * xy
     * x is a vector
     * y is a vector
     * alpha is a scala
     *
     * @param alpha
     * @param x
     * @param y
     * @return
     */
    public static double[] daxpy(double alpha, double[] x, double[] y) {
        double[] ret = new double[y.length];
        blas.dcopy(y.length, y, 0, ret, 0);
        blas.daxpy(x.length, alpha, x, 0, ret, 0);
        return y;
    }

    /**
     * Performs alpha * Ax + beta * y
     * A is a matrix, general format, column major
     * x is a vector
     * y is a vector
     * alpha and beta are scalar
     *
     * @param alpha
     * @param matrix
     * @param m
     * @param n
     * @param x
     * @param beta
     * @param y
     * @return
     */
    public static double[] dgemv(double alpha, double[] matrix, int m, int n, double[] x, double beta, double[] y) {
        double[] ret = new double[y.length];
        blas.dcopy(y.length, y, 0, ret, 0);
        blas.dgemv("N", m, n, alpha, matrix, m, x, 0, beta, ret, 0);
        return ret;
    }

    /**
     * Performs general matrix alpha * ab + beta * c
     * a, b, c are matrix
     * alpha, beta are scalar
     *
     * @param alpha
     * @param m
     * @param n
     * @param k
     * @param a
     * @param b
     * @param beta
     * @param c
     * @return
     */
    public static double[] dgemm(double alpha, int m, int n, int k, double[] a, double[] b, double beta, double[] c) {
        double[] ret = new double[c.length];
        blas.dcopy(c.length, c, 0, ret, 0);
        blas.dgemm("N", "N", m, n, k, alpha, a, m, b, k, beta, ret, m);
        return ret;
    }
}
