package org.haiyang.math.dl.gd;

import java.util.Collections;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

import org.haiyang.math.dl.networks.FeedForwardNetwork;
import org.haiyang.math.la.Matrix;
import org.haiyang.math.util.CollectionUtils;
import org.haiyang.math.dl.DeltaFunction;
import org.haiyang.math.dl.TrainingData;

import static java.util.stream.Collectors.toList;

/**
 * Stochastic gradient descent implementation
 */
public final class StochasticGradientDescent {
    private final List<TrainingData<Matrix, Matrix>> trainingData;
    private final int epochs;
    private final int miniBatchSize;
    private final double eta;
    private final double lamba;

    /**
     * Constructor to define a descent strategy.
     *
     * @param trainingData
     * @param epochs
     * @param miniBatchSize
     * @param eta
     */
    public StochasticGradientDescent(List<TrainingData<Matrix, Matrix>> trainingData, int epochs, int miniBatchSize,
            double eta, double lambda) {
        this.trainingData = trainingData;
        this.epochs = epochs;
        this.miniBatchSize = miniBatchSize;
        this.eta = eta;
        this.lamba = lambda;
    }

    /**
     * Actual descent work with training data.
     *
     * @param network
     * @param testData
     * @param evaluator
     * @param deltaFunc Function to compute teh delta value
     * @return
     */
    public FeedForwardNetwork descent(FeedForwardNetwork network, List<TrainingData<Matrix, Matrix>> testData,
                                      BiFunction<Matrix, Matrix, Boolean> evaluator, DeltaFunction deltaFunc) {
        int trainingSize = trainingData.size();
        int testDataSize = testData.size();
        for (int i = 0; i < epochs; i++) {
            long start = System.nanoTime();
            Collections.shuffle(trainingData);
            for (int b = 0; b < trainingSize; b += miniBatchSize) {
                network = descentMiniBatch(network, b, deltaFunc);
            }

            if (!testData.isEmpty()) {
                int res = network.evaluate(testData, evaluator);
                System.out.println(String.format("Epoch %d: %d / %d", i, res, testDataSize));
            }

            System.out.println(String.format("Epoch %d complete, time spent: %ds", i,
                    (System.nanoTime() - start) / 1_000_000_000L));
        }

        return network;
    }

    /**
     * Update the in's bias and weight using a batch starting at the offset
     * Return a new {@link FeedForwardNetwork} with updated biases and weights.
     *
     * @param in
     * @param trainingDataOffset
     * @return
     */
    private FeedForwardNetwork descentMiniBatch(FeedForwardNetwork in, int trainingDataOffset,
            DeltaFunction deltaFunc) {
        int totalSize = trainingData.size();
        List<Matrix> batchDeltaBiases = in.getBiases()
                                          .stream()
                                          .map(m -> new Matrix(m.getRowCount(), m.getColCount()))
                                          .collect(toList());
        List<Matrix> batchDeltaWeights = in.getWeights()
                                           .stream()
                                           .map(m -> new Matrix(m.getRowCount(), m.getColCount()))
                                           .collect(toList());
        FeedForwardNetwork batchResultNetwork = new FeedForwardNetwork(in.getLayerSizes(), batchDeltaBiases,
                batchDeltaWeights, in.getActivationFunction());
        int upperLimit = Math.min(trainingDataOffset + miniBatchSize, totalSize);
        batchResultNetwork = IntStream.range(trainingDataOffset, upperLimit)
                                      .parallel()
                                      .mapToObj(i -> {
                                          TrainingData<Matrix, Matrix> data = trainingData.get(i);
                                          return in.backprop(data.getX(), data.getY(), deltaFunc);
                                      })
                                      .reduce(batchResultNetwork, (a, b) -> {
                                          List<Matrix> bdb = CollectionUtils.zipApply(a.getBiases(), b.getBiases(),
                                                  (nb, db) -> nb.add(db));
                                          List<Matrix> bdw = CollectionUtils.zipApply(a.getWeights(), b.getWeights(),
                                                  (nw, bw) -> nw.add(bw));
                                          return new FeedForwardNetwork(in.getLayerSizes(), bdb, bdw,
                                                  in.getActivationFunction());
                                      });

        batchDeltaBiases = batchResultNetwork.getBiases();
        batchDeltaWeights = batchResultNetwork.getWeights();

        // now it is time to update the in
        List<Matrix> newWeights = CollectionUtils.zipApply(in.getWeights(), batchDeltaWeights,
                (w, bdw) -> w.mul(1 - eta * lamba / trainingData.size())
                             .minus(bdw.transform(v -> v * eta / miniBatchSize)));
        List<Matrix> newBiases = CollectionUtils.zipApply(in.getBiases(), batchDeltaBiases,
                (b, bdb) -> b.minus(bdb.transform(v -> v * eta / miniBatchSize)));

        return new FeedForwardNetwork(in.getLayerSizes(), newBiases, newWeights, in.getActivationFunction());
    }

}
