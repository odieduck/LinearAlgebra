package org.haiyang.math.dl.data;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.haiyang.math.la.Matrix;
import org.haiyang.math.dl.TrainingData;

/**
 * Given a binary data in byte array, parse them into a {@link List} of {@link TrainingData}
 */
public class MNISTDataLoader {
    /**
     * we have 10 digits to recognize
     */
    public static final int LABEL_TYPE_COUNT = 10;
    /**
     * The pixel values ranges from 0 to 255;
     */
    public static final double PIXEL_VALUE_MAX = 255.0;

    /**
     * Main method to test the loader
     *
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        String trainingDataPath = "/Volumes/Unix/LibearAlgebra/train-images-idx3-ubyte.gz";
        String trainingDataLabelsPath = "/Volumes/Unix/LibearAlgebra/train-labels-idx1-ubyte.gz";
        String testDataPath = "/Volumes/Unix/LibearAlgebra/t10k-images-idx3-ubyte.gz";
        String testDataLabelPath = "/Volumes/Unix/LibearAlgebra/t10k-labels-idx1-ubyte.gz";

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

        List<TrainingData<Matrix, Matrix>> trainingData = loadDataAsMatrix(trainingDataBytes, trainingDataLabelBytes);
        System.out.println("Done loading training data");
        List<TrainingData<Matrix, Matrix>> testData = loadDataAsMatrix(testDataBytes, testDataLabelBytes);
        System.out.println("Done loading testing data");
    }

    /**
     * Given the input binary data in the byte array, return a list of training data
     *
     * @param imageData
     * @param labelData
     * @return
     */
    public static List<TrainingData<Matrix, Matrix>> loadDataAsMatrix(ByteBuffer imageData, ByteBuffer labelData) {
        System.out.println("Data magic number (2051): " + imageData.getInt());
        System.out.println("Label magic number (2049): " + labelData.getInt());
        int itemCount = imageData.getInt();
        int labelcount = labelData.getInt();
        System.out.println("Number of images: " + itemCount);
        System.out.println("Number of labels: " + labelcount);
        int imageRows = imageData.getInt();
        int imageCols = imageData.getInt();
        System.out.println(String.format("Image has dimension of (%d, %d) (row, col)", imageRows, imageCols));
        // size of the 1-d array representing the image
        int dataLength = imageRows * imageCols;

        if (itemCount != labelcount) {
            throw new RuntimeException("Data item count must match label count");
        }

        // data to return
        List<TrainingData<Matrix, Matrix>> trainingData = new ArrayList<>(itemCount);
        for (int i = 0; i < itemCount; i++) {
            // read the buffer out
            byte[] buffer = new byte[dataLength];
            imageData.get(buffer);

            // convert into a double vector and matrix
            double[][] image = new double[dataLength][1];
            for (int k = 0; k < dataLength; k++) {
                // MNIST data has unsigned byte but java does not have unsigned byte
                // The work around is to cast to "int" and & 0xFF.
                // To normalize the value to (0,1) makes the modeling result very close to
                // http://neuralnetworksanddeeplearning.com/chap3.html
                image[k][0] = byteToUnsignedByte(buffer[k]) / PIXEL_VALUE_MAX;
            }

            // create the label matrix, marking the labeled element to 1
            double[][] label = new double[LABEL_TYPE_COUNT][1];
            label[labelData.get()][0] = 1;

            trainingData.add(new TrainingData<>(new Matrix(image), new Matrix(label)));
        }

        return trainingData;
    }

    /**
     * Convert a {@link byte} to unsigned byte (represented by an {@link int} since Java has no
     * unsigned byte
     *
     * @param b
     * @return
     */
    private static int byteToUnsignedByte(byte b) {
        return b & 0xFF;
    }
}
