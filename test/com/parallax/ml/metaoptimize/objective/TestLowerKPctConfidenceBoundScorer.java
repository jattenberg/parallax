/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.metaoptimize.objective;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Set;

import org.apache.commons.math.MathException;
import org.apache.commons.math.distribution.TDistribution;
import org.apache.commons.math.distribution.TDistributionImpl;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.junit.Test;

import com.google.common.collect.Sets;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.objective.AUCObjective;
import com.parallax.ml.objective.FoldEvaluator;
import com.parallax.ml.objective.LowerKPctConfidenceBoundScorer;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.testutils.TestUtils;

/**
 * The Class TestLower95ConfidenceBoundScorer.
 */
public class TestLowerKPctConfidenceBoundScorer {

	/** The dim. */
	int dim = (int) Math.pow(2, 14);

	/** The dummies. */
	Set<Double> dummies = Sets.newHashSet(1., 2., 3., 4., 5.);

	/**
	 * Test correct scores.
	 */
	@Test
	public void testCorrectScores() {
		AUCObjective obj = new AUCObjective();
		LowerKPctConfidenceBoundScorer<BinaryClassificationTarget> scorer = new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
				obj);
		for (double d : dummies) {
			scorer.evaluate(d);
		}

		assertEquals(1.492557, scorer.getScore(), 0.00001);
	}

	/**
	 * Test equal scores.
	 */

	@Test
	public void testEqualScores() {
		BinaryClassificationInstances insts = TestUtils.getTextInstances();
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				insts.getDimensions(), true);

		DescriptiveStatistics stats = new DescriptiveStatistics();
		AUCObjective obj = new AUCObjective();
		LowerKPctConfidenceBoundScorer<BinaryClassificationTarget> scorer = new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
				obj);
		int folds = 5;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = insts.getTraining(fold,
					folds);
			BinaryClassificationInstances testing = insts.getTesting(fold,
					folds);
			LogisticRegression model = builder.build();
			model.train(training);
			scorer.evaluate(testing, model);

			double score = obj.evaluate(testing, model);
			stats.addValue(score);
		}

		TDistribution tDist = new TDistributionImpl(stats.getN() - 1);
		double a = 0;
		try {
			a = tDist.inverseCumulativeProbability(0.95);
		} catch (MathException e) {
			throw new RuntimeException(e);
		}
		double width = a * stats.getStandardDeviation()
				/ Math.sqrt(stats.getN());
		double mean = stats.getMean();
		double val = mean - width;
		assertEquals(val, scorer.getScore(), 0.0001);
	}

	/**
	 * Test works multi threaded.
	 */
	@Test
	public void testWorksMultiThreaded() {
		BinaryClassificationInstances insts = TestUtils.getTextInstances();
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(
				insts.getDimensions(), true);

		AUCObjective obj = new AUCObjective();
		LowerKPctConfidenceBoundScorer<BinaryClassificationTarget> scorer = new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
				obj);

		int folds = 5;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances training = insts.getTraining(fold,
					folds);
			BinaryClassificationInstances testing = insts.getTesting(fold,
					folds);
			LogisticRegression model = builder.build();
			model.train(training);
			scorer.evaluate(testing, model);
		}
		double score = scorer.getScore();
		scorer = new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
				obj);
		FoldEvaluator eval = new FoldEvaluator(folds);

		double mtscore = eval.evaluate(insts,
				new LowerKPctConfidenceBoundScorer<BinaryClassificationTarget>(
						new AUCObjective()), builder);
		assertTrue(score > 0.5);
		assertTrue(mtscore > 0.5);
		assertEquals(score, mtscore, 0.0001);
	}
}
