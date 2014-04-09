package com.dsi.parallax.ml.objective;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.evaluation.ConfusionMatrix;
import com.dsi.parallax.ml.evaluation.ReceiverOperatingCharacteristic;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.objective.AUCObjective;
import com.dsi.parallax.ml.objective.AccuracyObjective;
import com.dsi.parallax.ml.objective.FoldEvaluator;
import com.dsi.parallax.ml.objective.MaxScorer;
import com.dsi.parallax.ml.objective.MinScorer;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.testutils.TestUtils;

public class TestFoldEvaluator {

	int folds = 10;

	@Test
	public void testFolds() throws InterruptedException {
		BinaryClassificationInstances instances = TestUtils.getTextInstances();
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				instances.getDimensions(), true);

		double mean = 0, meanACCY = 0;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = instances.getTraining(
					fold, folds);
			BinaryClassificationInstances testing = instances.getTesting(fold,
					folds);
			LogisticRegression model = builder.build();
			model.train(training);
			ReceiverOperatingCharacteristic ROC = new ReceiverOperatingCharacteristic();
			ConfusionMatrix conf = new ConfusionMatrix(2);
			for (BinaryClassificationInstance inst : testing) {
				BinaryClassificationTarget pred = model.predict(inst);
				ROC.add(inst.getLabel(), pred);
				conf.addInfo(inst.getLabel(), pred);
			}
			mean += ROC.binaryAUC();
			meanACCY += conf.computeAccuracy();
		}
		FoldEvaluator feval = new FoldEvaluator(folds);
		double mt = feval.evaluate(instances, new AUCObjective(), builder);
		double mtACCY = feval.evaluate(instances, new AccuracyObjective(),
				builder);
		assertEquals(mean / (double) folds, mt, 0.01);
		assertEquals(meanACCY / (double) folds, mtACCY, 0.01);

	}

	@Test
	public void testFoldsScorer() throws InterruptedException {
		BinaryClassificationInstances instances = TestUtils.getTextInstances();
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				instances.getDimensions(), true);

		double max = 0, min = Double.MAX_VALUE;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = instances.getTraining(
					fold, folds);
			BinaryClassificationInstances testing = instances.getTesting(fold,
					folds);
			LogisticRegression model = builder.build();
			model.train(training);
			ReceiverOperatingCharacteristic ROC = new ReceiverOperatingCharacteristic();
			for (BinaryClassificationInstance inst : testing)
				ROC.add(inst.getLabel(), model.predict(inst));

			max = Math.max(max, ROC.binaryAUC());
			min = Math.min(min, ROC.binaryAUC());
		}
		FoldEvaluator feval = new FoldEvaluator(folds);
		double maxMT = feval.evaluate(instances,
				new MaxScorer<BinaryClassificationTarget>(new AUCObjective()),
				builder);
		double minMT = feval.evaluate(instances,
				new MinScorer<BinaryClassificationTarget>(new AUCObjective()),
				builder);

		assertEquals(max, maxMT, 0.01);
		assertTrue(maxMT > 0.5);
		assertTrue(max > 0.5);
		assertEquals(min, minMT, 0.01);
		assertTrue(minMT > 0.5);
		assertTrue(min > 0.5);

	}

}
