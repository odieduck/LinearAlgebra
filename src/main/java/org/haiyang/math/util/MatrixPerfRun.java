package org.haiyang.math.util;

import java.util.Random;

import com.github.fommil.netlib.BLAS;
import org.haiyang.math.la.Matrix;

import static org.junit.Assert.assertEquals;

public class MatrixPerfRun {

    public static void main(String[] args) {
        int n = 1_000;
        double range = 1e3;
        int warmup = 100;
        System.out.println("warming up...");
        Matrix input = randMatrix(n, n, range);
        for (int i = 0; i < warmup; i++) {
            input = randMatrix(n, n, range);
        }

        System.out.println("start inversing");

        long start = System.nanoTime();

        Matrix inversed = input.inverse();

        long checkpoint = System.nanoTime();

        System.out.println("inverse time: " + ((checkpoint - start) / 1000000) + "ms");

        start = System.nanoTime();

        Matrix result = input.dot(inversed);

        checkpoint = System.nanoTime();

        System.out.println("mul time: " + ((checkpoint - start) / 1000000) + "ms");

        assertMatrixEquals(Matrix.identity(n), result);

        System.out.println("Verified!");

        // test out BLAS
        BLAS blas = BLAS.getInstance();
        double[] inArray = input.toArray();
        double[] invArray = inversed.toArray();
        double[] ret = new double[n * n];
        double[] expected = Matrix.identity(n).toArray();

        start = System.nanoTime();

        blas.dgemm("N", "N", n, n, n, 1.0, inArray, n, invArray, n, 0, ret, n);

        checkpoint = System.nanoTime();

        System.out.println("BLAS mul time: " + ((checkpoint - start) / 1000000) + "ms");
        for (int i = 0; i < ret.length; i++) {
            assertEquals(ret[i], expected[i], 1e-6);
        }
        System.out.println("BLAS verified!");
    }

    /**
     * assert if two matrix are the same.
     *
     * @param a
     * @param b
     * @param precision
     */
    public static void assertMatrixEquals(Matrix a, Matrix b, double precision) {
        assertEquals(a.getRowCount(), b.getRowCount());
        assertEquals(a.getColCount(), b.getColCount());

        for (int i = 0; i < a.getRowCount(); i++) {
            for (int j = 0; j < a.getColCount(); j++) {
                assertEquals(a.get(i, j), b.get(i, j), precision);
            }
        }
    }

    /**
     * Return a matrix with random values in it
     *
     * @param row
     * @param col
     * @return
     */
    public static Matrix randMatrix(int row, int col, double range) {
        double[][] data = new double[row][col];
        Random r = new Random();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = range * (r.nextDouble() - 0.5);
            }
        }

        return new Matrix(data);
    }

    public static Matrix randMatrix(int row, int col) {
        return randMatrix(row, col, 2 * 1e3);
    }

    /**
     * Default to 1e-6 precision.
     *
     * @param a
     * @param b
     */
    public static void assertMatrixEquals(Matrix a, Matrix b) {
        assertMatrixEquals(a, b, 1e-6);
    }

    /**
     * Print the matrix to command line
     */
    public static void print(Matrix m) {
        System.out.println();

        for (int i = 0; i < m.getRowCount(); i++) {
            for (int j = 0; j < m.getColCount(); j++) {
                System.out.format("%.3f ", m.get(i, j));
            }
            System.out.println();
        }

        System.out.println();
    }
}
