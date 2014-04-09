package com.dsi.parallax.ml.classifier.smoother;

import static org.junit.Assert.assertTrue;

import com.dsi.parallax.ml.classifier.linear.updateable.AROWClassifier;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.AROWClassifierBuilder;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.ScienceReader;

public class SmootherTestUtils {

	static int dim = (int) Math.pow(2, 14), min = 3;

	public static void testRegularization(SmootherType type) {
		AROWClassifierBuilder testBuilder = new AROWClassifierBuilder(dim, true)
				.setRegulizerType(type).setCrossvalidateSmootherTraining(
						5);
		AROWClassifierBuilder baselineBuilder = new AROWClassifierBuilder(dim,
				true);
		AROWClassifier baselineModel, testModel;

		BinaryClassificationInstances insts = ScienceReader.readScience(dim);

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances train = insts
					.getTraining(fold, folds);
			BinaryClassificationInstances test = insts.getTesting(fold, folds);
			assertTrue(train.getNumNeg() > 0);
			assertTrue(test.getNumNeg() > 0);
			assertTrue(train.getNumPos() > 0);
			assertTrue(train.getNumPos() > 0);

			testModel = testBuilder.build();
			baselineModel = baselineBuilder.build();

			OnlineEvaluation testEvaluation = new OnlineEvaluation();
			OnlineEvaluation baselineEvaluation = new OnlineEvaluation();

			testModel.train(train);
			baselineModel.train(train);

			for (BinaryClassificationInstance x : test) {
				double testPrediction = testModel.predict(x).getValue();
				double baselinePrediction = baselineModel.predict(x).getValue();
				double label = x.getLabel().getValue();

				testEvaluation.add(label, testPrediction);
				baselineEvaluation.add(label, baselinePrediction);
			}

			assertTrue(testEvaluation.computeAccuracy() > 0.5);
			assertTrue(testEvaluation.computeAUC() > 0.5);
			assertTrue(baselineEvaluation.computeAUC() > 0.5);
			assertTrue(testEvaluation.computeBrierScore() < baselineEvaluation
					.computeBrierScore());

		}
	}

	// @Test
	public static void testUpdateableRegularizer(SmootherType type) {
		AROWClassifierBuilder builder = new AROWClassifierBuilder(dim, true)
				.setRegulizerType(type).setCrossvalidateSmootherTraining(
						5);
		AROWClassifierBuilder builder2 = new AROWClassifierBuilder(dim, true);
		AROWClassifier model2, model;

		BinaryClassificationInstances insts = ScienceReader.readScience(dim);

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances train = insts
					.getTraining(fold, folds);
			BinaryClassificationInstances test = insts.getTesting(fold, folds);
			assertTrue(train.getNumNeg() > 0);
			assertTrue(test.getNumNeg() > 0);
			assertTrue(train.getNumPos() > 0);
			assertTrue(train.getNumPos() > 0);

			model = builder.build();
			model2 = builder2.build();

			OnlineEvaluation eval = new OnlineEvaluation();
			OnlineEvaluation eval2 = new OnlineEvaluation();

			for (BinaryClassificationInstance inst : train) {
				model.update(inst);
				model2.update(inst);
			}

			for (BinaryClassificationInstance x : test) {
				double pred = model.predict(x).getValue();
				double pred2 = model2.predict(x).getValue();
				double label = x.getLabel().getValue();

				eval.add(label, pred);
				eval2.add(label, pred2);
			}

			assertTrue(eval.computeAUC() > 0.5);
			assertTrue(eval2.computeAUC() > 0.5);
			assertTrue(eval.computeBrierScore() < eval2.computeBrierScore());

		}
	}
}
