/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.mercerkernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.kernelmethods.BudgetKernelPerceptron;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.BudgetKernelPerceptronBuilder;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestBudgetKernelPerceptron.
 */
public class TestBudgetKernelPerceptron {

	/** The dim. */
	int dim = (int) Math.pow(2, 18);

	/**
	 * Test.
	 */
	@Test
	public void test() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {

			BudgetKernelPerceptron model = new BudgetKernelPerceptron(dim, true);
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
	 * Test options.
	 */
	@Test
	public void testOptions() {
		BudgetKernelPerceptronBuilder builder = new BudgetKernelPerceptronBuilder(
				dim, false).setDecay(3.).setMargin(.12345).setPoolSize(17)
				.setWeight(1.2345).setPasses(23);

		BudgetKernelPerceptron model = builder.build();
		assertEquals(.12345, model.getMargin(), 0);
		assertEquals(17, model.getPoolSize(), 0);
		assertEquals(23, model.getPasses());
	}

	/**
	 * Test configuration.
	 */
	@Test
	public void testConfiguration() {
		BudgetKernelPerceptronBuilder builder = new BudgetKernelPerceptronBuilder(
				dim, false).setDecay(3.).setMargin(.12345).setPoolSize(50)
				.setWeight(1.2345).setPasses(2);

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		int folds = 2;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances train = insts
					.getTraining(fold, folds);
			BinaryClassificationInstances test = insts.getTesting(fold, folds);
			assertTrue(train.getNumNeg() > 0);
			assertTrue(test.getNumNeg() > 0);
			assertTrue(train.getNumPos() > 0);
			assertTrue(train.getNumPos() > 0);
			BudgetKernelPerceptron model = builder.build();
			OnlineEvaluation eval = new OnlineEvaluation();
			model.train(train);
			for (BinaryClassificationInstance x : test) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();

				eval.add(label, pred);
			}
			assertTrue(eval.computeAUC() > 0.5);
		}
	}
}
