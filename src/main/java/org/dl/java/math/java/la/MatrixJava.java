package org.dl.java.math.java.la;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

import static java.lang.System.arraycopy;

/**
 * This class represents the implementation of a matrix and its operations.
 * The implementation is pure java and is very slow compare to netlib-java
 */
public class MatrixJava {
    private final double[][] data;
    private final int row;
    private final int col;

    /**
     * Creating an all 0 matrix
     *
     * @param row
     * @param col
     */
    public MatrixJava(int row, int col) {
        if (row == 0 || col == 0) {
            throw new RuntimeException("Cannot create 0-dimension matrix!");
        }
        this.row = row;
        this.col = col;
        this.data = new double[row][col];
    }

    /**
     * Creating a {@link MatrixJava} from a 2-d array
     *
     * @param data
     */
    public MatrixJava(double[][] data) {
        if (data.length == 0 || data[0].length == 0) {
            throw new RuntimeException("Cannot create 0-dimension matrix!");
        }
        this.data = data;
        this.row = data.length;
        this.col = data[0].length;
    }

    /**
     * Creating a new {@link MatrixJava} instance from an existing instance
     *
     * @param matrixJava
     */
    public MatrixJava(MatrixJava matrixJava) {
        this.data = matrixJava.data;
        this.row = matrixJava.row;
        this.col = matrixJava.col;
    }

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
    public static MatrixJava getGaussionRandomMatrix(int row, int col, double mean, double std) {
        Random random = new Random();
        double[][] data = new double[row][col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                data[i][j] = random.nextGaussian() * std + mean;
            }
        }

        return new MatrixJava(data);
    }

    /**
     * Factory method to return a Gaussian random values matrix. The gaussian distribution is
     * with mean of 0 and variance of 1.
     *
     * @param row
     * @param col
     * @return
     */
    public static MatrixJava getGaussionRandomMatrix(int row, int col) {
        return getGaussionRandomMatrix(row, col, 0, 1);
    }

    public static MatrixJava identity(int n) {
        double[][] data = new double[n][n];
        for (int i = 0; i < n; i++) {
            data[i][i] = 1;
        }
        return new MatrixJava(data);
    }

    /**
     * Create a deep replicate;
     *
     * @return
     */
    public MatrixJava replicate() {
        double[][] data = new double[row][col];
        for (int i = 0; i < row; i++) {
            data[i] = Arrays.copyOf(this.data[i], col);
        }

        return new MatrixJava(data);
    }

    /**
     * Add a {@link MatrixJava} and create a new instance.
     *
     * @param in
     * @return
     */
    public MatrixJava add(MatrixJava in) {
        return add(in, true);
    }

    /**
     * Minus a {@link MatrixJava} and create a new instance.
     *
     * @param in
     * @return
     */
    public MatrixJava minus(MatrixJava in) {
        return add(in, false);
    }

    /**
     * Add the scalar value to each of the element of the matrix
     *
     * @param scalar
     * @return
     */
    public MatrixJava add(double scalar) {
        return transform(v -> v + scalar);
    }

    /**
     * Minus a scalar value
     *
     * @param scalar
     * @return
     */
    public MatrixJava minus(double scalar) {
        return add(-scalar);
    }

    /**
     * Multiply the scalar value to each element of the matrix
     *
     * @param scalar
     * @return
     */
    public MatrixJava mul(double scalar) {
        return transform(v -> v * scalar);
    }

    /**
     * Perform elements-wise multiplication. The two matrixJava must have the same dimension
     *
     * @param matrixJava
     * @return
     */
    public MatrixJava mul(MatrixJava matrixJava) {
        if (row == matrixJava.row && col == matrixJava.col) {
            double[][] res = new double[row][col];
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    res[i][j] = data[i][j] * matrixJava.data[i][j];
                }
            }

            return new MatrixJava(res);
        }

        throw new RuntimeException("Two matrixJava must have the same dimension for mul");
    }

    /**
     * product of this matrix to the right matrix.
     * <p>
     * Optimization note:
     * 1. Instead of ijk, we use ikj, such that dataIk can always be in the register, this gives 10x improvements.
     * 2. Using CPU's L3 caches, we did array copy such that the all inner loop computation is based on L3 cache,
     * this gives another 2x improvements on (1k, 1k) matrix multiplication.
     *
     * @param right
     * @return
     */
    public MatrixJava dot(MatrixJava right) {
        int rightRow = right.row;
        int rightCol = right.col;
        if (col == rightRow) {
            double[][] result = new double[row][rightCol];
            double[] rightBuffer = new double[rightCol];
            double[] resBuffer = new double[rightCol];
            for (int i = 0; i < row; i++) {
                for (int k = 0; k < col; k++) {
                    System.arraycopy(right.data[k], 0, rightBuffer, 0, rightCol);
                    double dataIk = data[i][k];
                    for (int j = 0; j < rightCol; j++) {
                        resBuffer[j] = Math.fma(dataIk, rightBuffer[j], resBuffer[j]);
                    }
                }
                System.arraycopy(resBuffer, 0, result[i], 0, rightCol);
                Arrays.fill(resBuffer, 0.0);
            }

            return new MatrixJava(result);
        }

        throw new RuntimeException(
                String.format("Dimension mismatch! left: (%d, %d), right: (%d, %d)", this.getRowCount(),
                        this.getColCount(), right.getRowCount(), right.getColCount()));
    }

    /**
     * Return the inverse of current matrix
     *
     * @return
     */
    public MatrixJava inverse() {
        if (row == col) {
            int n = data.length;
            LUDecomposition lu = lu();
            MatrixJava ret = null;
            for (int i = 0; i < n; i++) {
                double[][] b = new double[n][1];
                b[i][0] = 1;
                MatrixJava x = solve(new MatrixJava(b), lu);
                ret = ret == null ? x : ret.appendRight(x);
            }

            return ret;
        }

        throw new RuntimeException("Must be a square matrix!");
    }

    /**
     * Running the LU decomposition on the current matrix
     *
     * @return
     */
    public LUDecomposition lu() {
        int n = data.length;
        if (n == col) {
            MatrixJava u = (MatrixJava) replicate();
            MatrixJava l = identity(data.length);

            for (int k = 0; k < n - 1; k++) {
                for (int j = k + 1; j < n; j++) {
                    l.data[j][k] = u.data[j][k] / u.data[k][k];
                    for (int i = k; i < n; i++) {
                        u.data[j][i] -= l.data[j][k] * u.data[k][i];
                    }
                }
            }

            return new LUDecomposition(l, u);
        }

        throw new RuntimeException("LU decomposition can only be done to square matrix");
    }

    /**
     * Solve Ax = b
     *
     * @param b
     * @return
     */
    public MatrixJava solve(MatrixJava b) {
        return solve(b, lu());
    }

    /**
     * Append a matrix to the right: A.appendRight(B) means (A, B)
     * A and B must have the same row count
     *
     * @param right
     * @return
     */
    public MatrixJava appendRight(MatrixJava right) {
        if (row == right.row) {
            double[][] res = new double[row][col + right.col];
            for (int i = 0; i < row; i++) {
                arraycopy(data[i], 0, res[i], 0, col);
                arraycopy(right.data[i], 0, res[i], col, right.col);
            }

            return new MatrixJava(res);
        }

        throw new RuntimeException("matrix must have the same row count!");
    }

    /**
     * Append a matrix to the left.
     *
     * @param left
     * @return
     */
    public MatrixJava appendLeft(MatrixJava left) {
        return left.appendRight(this);
    }

    /**
     * Append a matrix to the bottom of the current matrix
     *
     * @param down
     * @return
     */
    public MatrixJava appendDown(MatrixJava down) {
        if (col == down.col) {
            double[][] res = new double[row + down.row][col];
            for (int i = 0; i < row; i++) {
                res[i] = Arrays.copyOf(data[i], col);
            }
            for (int i = 0; i < down.row; i++) {
                res[row + i] = Arrays.copyOf(down.data[i], down.col);
            }
            return new MatrixJava(res);
        }

        throw new RuntimeException("MatrixJava must have the same col count!");
    }

    /**
     * Append a matrix to the top of the current matrix
     *
     * @param up
     * @return
     */
    public MatrixJava appendUp(MatrixJava up) {
        return up.appendDown(this);
    }

    /**
     * Given a single element transformation function, apply to each element in the
     * matrix and generate a new one.
     *
     * @param transformer
     * @return
     */
    public MatrixJava transform(DoubleUnaryOperator transformer) {
        double[][] result = new double[row][col];

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                result[i][j] = transformer.applyAsDouble(data[i][j]);
            }
        }

        return new MatrixJava(result);
    }

    /**
     * Returns a transposed version of current matrix
     *
     * @return
     */
    public MatrixJava transpose() {
        double[][] res = new double[col][row];

        for (int i = 0; i < col; i++) {
            for (int j = 0; j < row; j++) {
                res[i][j] = data[j][i];
            }
        }

        return new MatrixJava(res);
    }

    /**
     * Returns the max index, if it is row vector, return the max row index, if it is a col vector,
     * return the max col index. Otherwise, throw {@link RuntimeException}
     *
     * @return
     */
    public int argmax() {
        if (row == 1) {
            return argmaxOfRow(0);
        } else if (col == 1) {
            return argmaxOfCol(0);
        }

        throw new RuntimeException("Does not support find max index on non-vector matrix");
    }

    /**
     * Given a column, find the max row index.
     *
     * @param col
     * @return
     */
    public int argmaxOfCol(int col) {
        return argmax(-1, col);

    }

    /**
     * Given a row, find the max column index
     *
     * @param row
     * @return
     */
    public int argmaxOfRow(int row) {
        return argmax(row, -1);
    }

    /**
     * Frobenius norm
     *
     * @return
     */
    public double norm() {
        double res = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                res += data[i][j] * data[i][j];
            }
        }
        return Math.sqrt(res);
    }

    /**
     * Return column-major 1-d array representing the matrix
     * The one-dimensional arrays in the exercises store the matrices by placing the elements of each column in successive cells of the arrays.
     *
     * @return
     */
    public double[] toArray() {
        double[] ret = new double[row * col];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                ret[i * col + j] = data[i][j];
            }
        }
        return ret;
    }

    /**
     * Return the value at (row, col)
     *
     * @param row
     * @param col
     * @return
     */
    public double get(int row, int col) {
        return data[row][col];
    }

    /**
     * Return row count
     *
     * @return
     */
    public int getRowCount() {
        return row;
    }

    /**
     * Return col count
     *
     * @return
     */
    public int getColCount() {
        return col;
    }

    /**
     * Supports both add and minus
     *
     * @param in
     * @param flag
     * @return
     */
    private MatrixJava add(MatrixJava in, boolean flag) {
        if (row == in.row && col == in.col) {
            double[][] result = new double[row][col];

            for (int i = 0; i < row; i++) {
                for (int j = 0; j < col; j++) {
                    result[i][j] = flag ? data[i][j] + in.data[i][j] : data[i][j] - in.data[i][j];
                }
            }

            return new MatrixJava(result);
        }

        throw new RuntimeException(
                String.format("Dimension mismatch! This: (%d, %d), in: (%d, %d)", row, col, in.row, in.col));
    }

    /**
     * Solve Ax = b, but pass in a precomputed LU
     * The LU must be the LU decomposition of current matrix.
     * <p>
     * If there are more than 1 column of b, then solve for every Column
     *
     * @param b
     * @param lu
     * @return
     */
    private MatrixJava solve(MatrixJava b, LUDecomposition lu) {
        if (b.row == row && b.col == 1) {
            int n = data.length;

            // forward substitution, solving ld = b
            MatrixJava d = new MatrixJava(new double[n][1]);
            for (int i = 0; i < n; i++) {
                double sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += lu.l()
                            .get(i, j) * d.data[j][0];
                }

                d.data[i][0] = b.data[i][0] - sum;
            }

            // back propagation, solving Ux = d
            MatrixJava x = new MatrixJava(new double[n][1]);
            for (int i = n - 1; i >= 0; i--) {
                double sum = 0;
                for (int j = i + 1; j < n; j++) {
                    sum += lu.u()
                            .get(i, j) * x.data[j][0];
                }
                x.data[i][0] = (d.data[i][0] - sum) / lu.u()
                        .get(i, i);
            }

            return x;
        }

        throw new RuntimeException("Dimension mismatch!");
    }

    /**
     * Find max value index in either a row or a column
     *
     * @param row
     * @param col
     * @return
     */
    private int argmax(int row, int col) {
        if (row != -1 && col != -1) {
            throw new RuntimeException("Cannot find argmax for both row and col!");
        }

        boolean isRowSet = row != -1;
        int limit = isRowSet ? this.col : this.row;
        int ind = 0;
        double max = Double.MIN_VALUE;
        for (int n = 0; n < limit; n++) {
            double value = isRowSet ? data[row][n] : data[n][col];
            if (value > max) {
                max = value;
                ind = n;
            }
        }

        return ind;
    }
}
