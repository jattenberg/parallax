package com.dsi.parallax.ml.examples.text;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.ScienceReader;

public class FindBestModel {

	public static void main(String[] args) {
		BinaryClassificationInstances insts = ScienceReader.readScience();
		int folds = 10;

		for (Classifiers classifier : Classifiers.values()) {
			Classifier<?> model = classifier.getClassifier(
					ScienceReader.DIMENSION, true);
			OnlineEvaluation eval = new OnlineEvaluation();
			System.out.print("training a " + classifier + " on fold:");
			for (int fold = 0; fold < folds; fold++) {
				System.out.print(" " + fold);
				BinaryClassificationInstances training = insts.getTraining(
						fold, folds);
				BinaryClassificationInstances testing = insts.getTesting(fold,
						folds);
				model.train(training);
				for (BinaryClassificationInstance inst : testing) {
					eval.add(inst.getLabel(), model.predict(inst));
				}

			}
			System.out.println("\nperformance for: " + classifier);
			System.out.println(eval);
		}
	}
}
