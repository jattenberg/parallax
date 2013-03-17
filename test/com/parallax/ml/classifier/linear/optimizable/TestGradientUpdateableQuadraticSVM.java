package com.parallax.ml.classifier.linear.optimizable;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder.GradientUpdateableQuadraticSVMBuilder;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableTestUtils;
import com.parallax.ml.evaluation.OnlineEvaluation;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.utils.IrisReader;
import com.parallax.optimization.stochastic.SGDBuilder;
import com.parallax.optimization.stochastic.StochasticBFGSBuilder;
import com.parallax.optimization.stochastic.StochasticLBFGSBuilder;
import com.parallax.optimization.stochastic.anneal.AnnealingScheduleConfigurableBuilder;
import com.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

public class TestGradientUpdateableQuadraticSVM {

	/** The dim. */
	int dim = (int) Math.pow(2, 18);
	int smallDim = 4;
	int folds = 3;

	/** The sgd builder. */
	SGDBuilder sgdBuilder = new SGDBuilder(dim, true);
	StochasticBFGSBuilder bfgsBuilder = new StochasticBFGSBuilder(smallDim,
			true);
	StochasticLBFGSBuilder lbfgsBuilder = new StochasticLBFGSBuilder(smallDim,
			true);

	/**
	 * Test SGD updating.
	 */
	@Test
	public void testSGD() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next().shuffle();

		sgdBuilder
				.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder
						.configureForConstantRate(0.01));

		for (int fold = 0; fold < folds; fold++) {
			GradienUpdateableQuadraticSVM model = new GradienUpdateableQuadraticSVM(
					sgdBuilder, dim, true);
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

	/**
	 * Test vary gamma.
	 */
	@Test
	public void testVaryingGamma() {
		BinaryClassificationInstances insts = IrisReader.readIris().shuffle();

		for (double gamma = 0.0000001; gamma < 10; gamma *= 50) {
			sgdBuilder
					.setAnnealingScheduleConfigurableBuilder(AnnealingScheduleConfigurableBuilder
							.configureForConstantRate(0.01));
			for (int fold = 0; fold < folds; fold++) {
				GradienUpdateableQuadraticSVM model = new GradienUpdateableQuadraticSVM(
						sgdBuilder, dim, true).setGamma(gamma);
				OnlineEvaluation eval = new OnlineEvaluation();

				model.train(insts.getTraining(fold, folds));
				for (BinaryClassificationInstance x : insts.getTesting(fold,
						folds)) {
					double pred = model.predict(x).getValue();
					double label = x.getLabel().getValue();
					eval.add(label, pred);
				}
				assertTrue(eval.computeAUC() > 0.5);
			}
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

			GradienUpdateableQuadraticSVM model = new GradienUpdateableQuadraticSVM(
					bfgsBuilder, smallDim, true);
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
			GradienUpdateableQuadraticSVM model = new GradienUpdateableQuadraticSVM(
					lbfgsBuilder, smallDim, true);
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
				.testTruncationOptimizable(new GradientUpdateableQuadraticSVMBuilder());
	}
}
