package com.dsi.parallax.ml.classifier.ensemble;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.ensemble.BaggingEnsenbleClassifier;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.AROWClassifierBuilder;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.IrisReader;

public class TestBaggingEnsenbleClassifier {

	/**
	 * Test.
	 */
	@Test
	public void test() {
		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {

			BinaryClassificationInstances insts = IrisReader.readIris();
			BaggingEnsenbleClassifier model = new BaggingEnsenbleClassifier(
					IrisReader.DIMENSION, false).initialize();
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts
					.getTraining(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
			assertTrue(eval.computeAccuracy() > 0.5);
		}
	}

	@Test
	public void testBias() {
		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {

			BinaryClassificationInstances insts = IrisReader.readIris();
			BaggingEnsenbleClassifier model = new BaggingEnsenbleClassifier(
					IrisReader.DIMENSION, true).initialize();
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts
					.getTraining(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
			assertTrue(eval.computeAccuracy() > 0.5);
		}
	}

	@Test
	public void testNewClassifier() {
		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {

			BinaryClassificationInstances insts = IrisReader.readIris();
			BaggingEnsenbleClassifier model = new BaggingEnsenbleClassifier(
					IrisReader.DIMENSION, true)
					.setNumModels(10)
					.setClassifierBuilder(
							new AROWClassifierBuilder(IrisReader.DIMENSION,
									true)
									.setRegulizerType(SmootherType.PLATT))
					.initialize();
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts
					.getTraining(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			System.out.println(eval);
			assertTrue(eval.computeAUC() > 0.5);
			assertTrue(eval.computeAccuracy() > 0.5);
		}
	}

}
