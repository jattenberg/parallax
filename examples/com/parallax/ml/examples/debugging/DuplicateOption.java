package com.parallax.ml.examples.debugging;

import java.util.Arrays;

import com.parallax.ml.classifier.linear.optimizable.GradienUpdateableQuadraticSVM;
import com.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableQuadraticSVMBuilder;
import com.parallax.ml.classifier.smoother.SmootherType;
import com.parallax.ml.evaluation.ConfusionMatrix;
import com.parallax.ml.evaluation.ReceiverOperatingCharacteristic;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.utils.IrisReader;

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
			GradienUpdateableQuadraticSVM model = builder.build();
			// model = (GradienUpdateableQuadraticSVM)builder.build();

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
