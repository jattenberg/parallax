/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.optimizable;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableL2;
import com.dsi.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableL2Builder;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableTestUtils;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.IrisReader;
import com.dsi.parallax.optimization.stochastic.SGDBuilder;
import com.dsi.parallax.optimization.stochastic.StochasticBFGSBuilder;
import com.dsi.parallax.optimization.stochastic.StochasticLBFGSBuilder;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestGradientUpdateableL2.
 */
public class TestGradientUpdateableL2 {

	/** The dim. */
	int dim = (int) Math.pow(2, 18);
	int smallDim = 4;

	/** The sgd builder. */
	SGDBuilder sgdBuilder = new SGDBuilder(dim, true);
	StochasticBFGSBuilder bfgsBuilder = new StochasticBFGSBuilder(smallDim,
			true);
	StochasticLBFGSBuilder lbfgsBuilder = new StochasticLBFGSBuilder(smallDim,
			true);

	/**
	 * Test.
	 */
	@Test
	public void test() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next().shuffle();

		sgdBuilder
				.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder
						.configureForConstantRate(0.01));
		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			GradientUpdateableL2 model = new GradientUpdateableL2(sgdBuilder,
					dim, true);
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts.getTesting(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
		}
	}

	@Test
	public void testBFGS() {
		BinaryClassificationInstances insts = IrisReader.readIris().shuffle();

		bfgsBuilder
				.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder
						.configureForConstantRate(0.01));
		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {

			GradientUpdateableL2 model = new GradientUpdateableL2(bfgsBuilder,
					smallDim, true);
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts.getTesting(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
		}
	}

	@Test
	public void testLBFGS() {
		BinaryClassificationInstances insts = IrisReader.readIris().shuffle();

		lbfgsBuilder
				.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder
						.configureForConstantRate(0.01));
		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			GradientUpdateableL2 model = new GradientUpdateableL2(lbfgsBuilder,
					smallDim, true);
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts.getTesting(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();

				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
		}
	}

	@Test
	public void testTruncation() {
		LinearUpdateableTestUtils
				.testTruncationOptimizable(new GradientUpdateableL2Builder());
	}
}
