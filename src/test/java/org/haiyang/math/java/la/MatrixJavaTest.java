package org.haiyang.math.java.la;

import org.haiyang.math.java.util.MatrixPerfRun;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.haiyang.math.java.util.MatrixPerfRun.assertMatrixEquals;
import static org.haiyang.math.java.util.MatrixPerfRun.randMatrix;

public class MatrixJavaTest {

    @Test(expected = RuntimeException.class)
    public void testAddMatrix() {
        MatrixJava one = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        MatrixJava two = new MatrixJava(new double[][] { { 6.0, 5.0, 4.0 }, { 3.0, 2.0, 1.0 } });
        MatrixJava expected = new MatrixJava(new double[][] { { 7.0, 7.0, 7.0 }, { 7.0, 7.0, 7.0 } });
        MatrixJava mismatch = new MatrixJava(new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        MatrixJava result = one.add(two);
        MatrixPerfRun.assertMatrixEquals(result, expected);
        one.add(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testMultiply() {
        MatrixJava one = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        MatrixJava two = new MatrixJava(new double[][] { { 6.0, 5.0 }, { 4.0, 3.0 }, { 2.0, 1.0 } });
        MatrixJava mismatch = new MatrixJava(new double[][] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        MatrixJava rightExpected = new MatrixJava(new double[][] { { 20.0, 14.0 }, { 56.0, 41.0 } });
        MatrixJava leftExpected = new MatrixJava(
                new double[][] { { 26.0, 37.0, 48.0 }, { 16.0, 23.0, 30.0 }, { 6.0, 9.0, 12.0 } });

        // test dot product
        MatrixJava result = one.dot(two);
        MatrixPerfRun.assertMatrixEquals(rightExpected, result);

        one.dot(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendRight() {
        MatrixJava one = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        MatrixJava two = new MatrixJava(new double[][] { { 1.0 }, { 2.0 } });
        MatrixJava expected = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0, 1.0 }, { 4.0, 5.0, 6.0, 2.0 } });
        MatrixJava mismatch = new MatrixJava(new double[][] { { 1.0 } });

        MatrixJava result = one.appendRight(two);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendRight(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendLeft() {
        MatrixJava one = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        MatrixJava two = new MatrixJava(new double[][] { { 1.0 }, { 2.0 } });
        MatrixJava expected = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0, 1.0 }, { 4.0, 5.0, 6.0, 2.0 } });
        MatrixJava mismatch = new MatrixJava(new double[][] { { 1.0 } });

        MatrixJava result = two.appendLeft(one);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendRight(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendDown() {
        MatrixJava one = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        MatrixJava two = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 } });
        MatrixJava expected = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 1.0, 2.0, 3.0 } });
        MatrixJava mismatch = new MatrixJava(new double[][] { { 1.0 } });

        MatrixJava result = one.appendDown(two);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendDown(mismatch);
    }

    @Test(expected = RuntimeException.class)
    public void testAppendUp() {
        MatrixJava one = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 } });
        MatrixJava two = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 } });
        MatrixJava expected = new MatrixJava(new double[][] { { 1.0, 2.0, 3.0 }, { 4.0, 5.0, 6.0 }, { 1.0, 2.0, 3.0 } });
        MatrixJava mismatch = new MatrixJava(new double[][] { { 1.0 } });

        MatrixJava result = two.appendUp(one);
        MatrixPerfRun.assertMatrixEquals(result, expected);

        one.appendUp(mismatch);
    }

    @Test
    public void testLUDecomposition() {
        MatrixJava in = new MatrixJava(new double[][] { { 1, 1, 1 }, { 2, 3, 5 }, { 4, 6, 8 } });
        LUDecomposition lu = in.lu();
        MatrixJava res = lu.l()
                       .dot(lu.u());

        MatrixPerfRun.assertMatrixEquals(res, in);

        MatrixJava rand = MatrixPerfRun.randMatrix(100, 100);
        lu = rand.lu();
        MatrixPerfRun.assertMatrixEquals(rand, lu.l()
                                   .dot(lu.u()));
    }

    @Test
    public void testInverse() {
        MatrixJava in = new MatrixJava(new double[][] { { 1, 0, 2 }, { 2, -1, 3 }, { 4, 1, 8 } });
        MatrixJava inversed = in.inverse();
        MatrixPerfRun.assertMatrixEquals(MatrixJava.identity(in.getColCount()), in.dot(inversed));

        int n = 100;
        MatrixJava rand = MatrixPerfRun.randMatrix(n, n);
        MatrixPerfRun.assertMatrixEquals(MatrixJava.identity(n), rand.dot(rand.inverse()));
        MatrixPerfRun.assertMatrixEquals(MatrixJava.identity(n), rand.inverse()
                                                   .dot(rand));
    }

    @Test
    public void testSolve() {
        int n = 10;
        MatrixJava A = MatrixPerfRun.randMatrix(n, n);
        MatrixJava b = MatrixPerfRun.randMatrix(n, 1);
        MatrixJava x = A.solve(b);

        MatrixPerfRun.assertMatrixEquals(A.dot(x), b);
    }

    @Test
    public void testTranspose() {
        MatrixJava a = MatrixPerfRun.randMatrix(100, 100);
        MatrixPerfRun.assertMatrixEquals(a, a.transpose()
                               .transpose());
    }

    @Test
    public void testArgmaxRow() {
        MatrixJava in = new MatrixJava(new double[][] { { 1, 0, 2 }, { 2, -1, 3 }, { 4, 1, 8 } });
        assertEquals(2, in.argmaxOfCol(0));
        assertEquals(2, in.argmaxOfCol(1));
        assertEquals(2, in.argmaxOfCol(2));
    }

    @Test
    public void testArgmaxCol() {
        MatrixJava in = new MatrixJava(new double[][] { { 1, 0, 2 }, { 2, -1, 3 }, { 4, 1, 8 } });
        assertEquals(2, in.argmaxOfRow(0));
        assertEquals(2, in.argmaxOfRow(1));
        assertEquals(2, in.argmaxOfRow(2));
    }

    @Test
    public void testArgmax() {
        MatrixJava m = new MatrixJava(new double[][] { { 1, 2, 3, 10, 5, 6 } });
        assertEquals(3, m.argmax());
        m = m.transpose();
        assertEquals(3, m.argmax());
    }

    @Test
    public void testAddScalar() {
        MatrixJava m = new MatrixJava(new double[][] { { 1, 2, 3, 10, 5, 6 } });
        MatrixJava res = m.add((2.0));
        MatrixJava expected = new MatrixJava(new double[][] { { 3, 4, 5, 12, 7, 8 } });
        MatrixPerfRun.assertMatrixEquals(res, expected);
    }

    @Test
    public void testMulScalar() {
        MatrixJava m = new MatrixJava(new double[][] { { 1, 2, 3, 10, 5, 6 } });
        MatrixJava res = m.mul((2.0));
        MatrixJava expected = new MatrixJava(new double[][] { { 2, 4, 6, 20, 10, 12 } });
    }

    @Test
    public void testElementsMul() {
        MatrixJava one = new MatrixJava(new double[][] { { 1, 2, 3 }, { 10, 5, 6 } });
        MatrixJava two = new MatrixJava(new double[][] { { 11, 12, 13 }, { 2, 5, 5 } });
        MatrixJava expected = new MatrixJava(new double[][] { { 11, 24, 39 }, { 20, 25, 30 } });
        MatrixPerfRun.assertMatrixEquals(one.mul(two), expected);
        MatrixPerfRun.assertMatrixEquals(two.mul(one), expected);
    }
}
