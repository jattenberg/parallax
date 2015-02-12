package com.dsi.parallax.ml.examples.debugging;

import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableQuadraticSVM;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableQuadraticSVMBuilder;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableQuadraticSVM;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.evaluation.ConfusionMatrix;
import com.dsi.parallax.ml.evaluation.ReceiverOperatingCharacteristic;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.IrisReader;

import java.util.Arrays;

public class DuplicateOption {
	public static void main(String[] args) {
		BinaryClassificationInstances instances = IrisReader.readIris();
		GradientUpdateableQuadraticSVMBuilder builder = new GradientUpdateableQuadraticSVMBuilder(
				IrisReader.DIMENSION, true);
		builder.setCrossvalidateSmootherTraining(3).setMiniBatchSize(5)
				.setRegulizerType(SmootherType.ISOTONIC).setPasses(5)
				.setWeight(3).setGamma(2.0);

		System.out.println(Arrays.toString(builder
				.getArgumentsFromConfiguration()));

		ConfusionMatrix conf = new ConfusionMatrix(2);
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();

		boolean verbose = true;
		int folds = 10;

		// Performing an n-fold cross-validation. In this case, n has been
		// initialized to 10.
		for (int fold = 0; fold < folds; fold++) {
			if (verbose) {
				System.out.println("on fold: " + (fold + 1));
			}

			// Functions for obtaining training and testing set while
			// cross-validation
			BinaryClassificationInstances training = instances.getTraining(
					fold, folds);
			BinaryClassificationInstances testing = instances.getTesting(fold,
					folds);

			// Initializing the classifier function
			GradientUpdateableQuadraticSVM model = builder.build();
			// model = (GradientUpdateableQuadraticSVM)builder.build();

			// Training of the model
			model.train(training);

			// Validating the model built using the training set, with
			// the testing sample
			for (BinaryClassificationInstance instance : testing) {
				double label = instance.getLabel().getValue();

				// Function for predicting the value using the model built
				// with the training data
				double prediction = model.predict(instance).getValue();
				conf.addInfo(label, prediction);
				roc.add(label, prediction);
			}
			// System.out.println(((AbstractLinearUpdateableClassifier<AROWClassifier>)
			// model).getVector());
		}

		System.out.println(conf);
		System.out.println(conf.getIR());
		System.out.println("AUC: " + roc.binaryAUC());
	}
}
