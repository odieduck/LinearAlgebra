package org.dl.java.math.java.util;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.dl.java.math.java.dl.DeltaFunctionJava;
import org.dl.java.io.data.TrainingData;
import org.dl.java.math.java.dl.gd.StochasticGradientDescent;
import org.dl.java.math.java.dl.networks.FeedForwardNeuralNetwork;
import org.dl.java.math.java.la.MatrixJava;

import static org.dl.java.math.java.dl.activate.ActivateFunctionsJava.SIGMOID;
import static org.dl.java.io.data.MNISTDataLoader.loadDataAsMatrix;

/**
 * Running simple FFN for hand written digits recognition
 */
public class MNISTDataRun {
    /**
     * Represents the input layer
     */
    public static final int IMAGE_DATA_LENGTH = 784;

    public static void main(String[] args) throws IOException {
        String trainingDataPath = "./train-images-idx3-ubyte.gz";
        String trainingDataLabelsPath = "./train-labels-idx1-ubyte.gz";
        String testDataPath = "./t10k-images-idx3-ubyte.gz";
        String testDataLabelPath = "./t10k-labels-idx1-ubyte.gz";

        ByteBuffer trainingDataBytes = ByteBuffer.wrap(
                new GZIPInputStream(new FileInputStream(trainingDataPath)).readAllBytes());
        ByteBuffer trainingDataLabelBytes = ByteBuffer.wrap(
                new GZIPInputStream(new FileInputStream(trainingDataLabelsPath)).readAllBytes());
        ByteBuffer testDataBytes = ByteBuffer.wrap(
                new GZIPInputStream(new FileInputStream(testDataPath)).readAllBytes());
        ByteBuffer testDataLabelBytes = ByteBuffer.wrap(
                new GZIPInputStream(new FileInputStream(testDataLabelPath)).readAllBytes());

        // verify data unzipped
        System.out.println(trainingDataBytes.capacity());
        System.out.println(trainingDataLabelBytes.capacity());
        System.out.println(testDataBytes.capacity());
        System.out.println(testDataLabelBytes.capacity());

        List<TrainingData<MatrixJava, MatrixJava>> trainingData = loadDataAsMatrix(trainingDataBytes, trainingDataLabelBytes);
        System.out.println("Done loading training data");
        List<TrainingData<MatrixJava, MatrixJava>> testData = loadDataAsMatrix(testDataBytes, testDataLabelBytes);
        System.out.println("Done loading testing data");

        // Creating a new network
        FeedForwardNeuralNetwork ffn = new FeedForwardNeuralNetwork(Arrays.asList(IMAGE_DATA_LENGTH, 30, 10), SIGMOID);

        // Creating the strategy to gradient descent
        // 30 runs
        int epochs = 30;
        // mini batch size
        int miniBatchSize = 10;
        // training rate
        double eta = 0.1;
        double lambda = 5;
        StochasticGradientDescent sgd = new StochasticGradientDescent(trainingData, epochs, miniBatchSize, eta, lambda);
        // Using quadratic cost function
        //                ffn = sgd.descent(ffn, trainingData.subList(testCount, allCount), (x, y) -> x.argmax() == y.argmax(),
        //                        DeltaFunctionJava.QUADRATIC);
        // Using cross-entropy cost function
        ffn = sgd.descent(ffn, testData, (x, y) -> x.argmax() == y.argmax(), DeltaFunctionJava.CROSS_ENTROPY);
    }
}
