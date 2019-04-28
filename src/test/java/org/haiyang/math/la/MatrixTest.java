package org.haiyang.math.la;

import org.haiyang.math.util.MatrixPerfRun;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.haiyang.math.util.MatrixPerfRun.assertMatrixEquals;
import static org.haiyang.math.util.MatrixPerfRun.randMatrix;

public class MatrixTest {

    @Test(expected = RuntimeException.class)
    public void testAddMatrix() {
        Matrix one = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        Matrix two = new Matrix(new double[][] { { 6.0, 5.0, 4.0 }, { 3.0, 2.0, 1.0 } });
        Matrix expected = new Matrix(new double[][] { { 7.0, 7.0, 7.0 }, { 7.0, 7.0, 7.0 } });
        Matrix mismatch = new Matrix(new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        Matrix result = one.add(two);
        MatrixPerfRun.assertMatrixEquals(result, expected);
        one.add(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testMultiply() {
        Matrix one = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        Matrix two = new Matrix(new double[][] { { 6.0, 5.0 }, { 4.0, 3.0 }, { 2.0, 1.0 } });
        Matrix mismatch = new Matrix(new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        Matrix rightExpected = new Matrix(new double[][] { { 20.0, 14.0 }, { 56.0, 41.0 } });
        Matrix leftExpected = new Matrix(
                new double[][] { { 26.0, 37.0, 48.0 }, { 16.0, 23.0, 30.0 }, { 6.0, 9.0, 12.0 } });

        // test dot product
        Matrix result = one.dot(two);
        MatrixPerfRun.assertMatrixEquals(rightExpected, result);

        one.dot(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendRight() {
        Matrix one = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        Matrix two = new Matrix(new double[][] { { 1.0 }, { 2.0 } });
        Matrix expected = new Matrix(new double[][] { { 1.0, 2.0, 3.0, 1.0 }, { 4.0, 5.0, 6.0, 2.0 } });
        Matrix mismatch = new Matrix(new double[][] { { 1.0 } });

        Matrix result = one.appendRight(two);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendRight(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendLeft() {
        Matrix one = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        Matrix two = new Matrix(new double[][] { { 1.0 }, { 2.0 } });
        Matrix expected = new Matrix(new double[][] { { 1.0, 2.0, 3.0, 1.0 }, { 4.0, 5.0, 6.0, 2.0 } });
        Matrix mismatch = new Matrix(new double[][] { { 1.0 } });

        Matrix result = two.appendLeft(one);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendRight(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendDown() {
        Matrix one = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        Matrix two = new Matrix(new double[][] { { 1.0, 2.0, 3.0 } });
        Matrix expected = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 1.0, 2.0, 3.0 } });
        Matrix mismatch = new Matrix(new double[][] { { 1.0 } });

        Matrix result = one.appendDown(two);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendDown(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendUp() {
        Matrix one = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        Matrix two = new Matrix(new double[][] { { 1.0, 2.0, 3.0 } });
        Matrix expected = new Matrix(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 1.0, 2.0, 3.0 } });
        Matrix mismatch = new Matrix(new double[][] { { 1.0 } });

        Matrix result = two.appendUp(one);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendUp(mismatch);
    }

    @Test
    public void testLUDecomposition() {
        Matrix in = new Matrix(new double[][] { { 1, 1, 1 }, { 2, 3, 5 }, { 4, 6, 8 } });
        LUDecomposition lu = in.lu();
        Matrix res = lu.l()
                       .dot(lu.u());

        MatrixPerfRun.assertMatrixEquals(res, in);

        Matrix rand = MatrixPerfRun.randMatrix(100, 100);
        lu = rand.lu();
        MatrixPerfRun.assertMatrixEquals(rand, lu.l()
                                   .dot(lu.u()));
    }

    @Test
    public void testInverse() {
        Matrix in = new Matrix(new double[][] { { 1, 0, 2 }, { 2, -1, 3 }, { 4, 1, 8 } });
        Matrix inversed = in.inverse();
        MatrixPerfRun.assertMatrixEquals(Matrix.identity(in.getColCount()), in.dot(inversed));

        int n = 100;
        Matrix rand = MatrixPerfRun.randMatrix(n, n);
        MatrixPerfRun.assertMatrixEquals(Matrix.identity(n), rand.dot(rand.inverse()));
        MatrixPerfRun.assertMatrixEquals(Matrix.identity(n), rand.inverse()
                                                   .dot(rand));
    }

    @Test
    public void testSolve() {
        int n = 10;
        Matrix A = MatrixPerfRun.randMatrix(n, n);
        Matrix b = MatrixPerfRun.randMatrix(n, 1);
        Matrix x = A.solve(b);

        MatrixPerfRun.assertMatrixEquals(A.dot(x), b);
    }

    @Test
    public void testTranspose() {
        Matrix a = MatrixPerfRun.randMatrix(100, 100);
        MatrixPerfRun.assertMatrixEquals(a, a.transpose()
                               .transpose());
    }

    @Test
    public void testArgmaxRow() {
        Matrix in = new Matrix(new double[][] { { 1, 0, 2 }, { 2, -1, 3 }, { 4, 1, 8 } });
        assertEquals(2, in.argmaxOfCol(0));
        assertEquals(2, in.argmaxOfCol(1));
        assertEquals(2, in.argmaxOfCol(2));
    }

    @Test
    public void testArgmaxCol() {
        Matrix in = new Matrix(new double[][] { { 1, 0, 2 }, { 2, -1, 3 }, { 4, 1, 8 } });
        assertEquals(2, in.argmaxOfRow(0));
        assertEquals(2, in.argmaxOfRow(1));
        assertEquals(2, in.argmaxOfRow(2));
    }

    @Test
    public void testArgmax() {
        Matrix m = new Matrix(new double[][] { { 1, 2, 3, 10, 5, 6 } });
        assertEquals(3, m.argmax());
        m = m.transpose();
        assertEquals(3, m.argmax());
    }

    @Test
    public void testAddScalar() {
        Matrix m = new Matrix(new double[][] { { 1, 2, 3, 10, 5, 6 } });
        Matrix res = m.add((2.0));
        Matrix expected = new Matrix(new double[][] { { 3, 4, 5, 12, 7, 8 } });
        MatrixPerfRun.assertMatrixEquals(res, expected);
    }

    @Test
    public void testMulScalar() {
        Matrix m = new Matrix(new double[][] { { 1, 2, 3, 10, 5, 6 } });
        Matrix res = m.mul((2.0));
        Matrix expected = new Matrix(new double[][] { { 2, 4, 6, 20, 10, 12 } });
    }

    @Test
    public void testElementsMul() {
        Matrix one = new Matrix(new double[][] { { 1, 2, 3 }, { 10, 5, 6 } });
        Matrix two = new Matrix(new double[][] { { 11, 12, 13 }, { 2, 5, 5 } });
        Matrix expected = new Matrix(new double[][] { { 11, 24, 39 }, { 20, 25, 30 } });
        MatrixPerfRun.assertMatrixEquals(one.mul(two), expected);
        MatrixPerfRun.assertMatrixEquals(two.mul(one), expected);
    }
}
