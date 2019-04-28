package org.haiyang.math.dl;

/**
 * A generic wrapper class for training data
 *
 * @param <X>
 * @param <Y>
 */
public final class TrainingData<X, Y> {
    private final X x;
    private final Y y;

    /**
     * Create a training data point
     *
     * @param x
     * @param y
     */
    public TrainingData(X x, Y y) {
        this.x = x;
        this.y = y;
    }

    /**
     * Get the training input
     *
     * @return
     */
    public X getX() {
        return x;
    }

    /**
     * Get the training output
     *
     * @return
     */
    public Y getY() {
        return y;
    }
}
