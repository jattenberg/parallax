/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.mercerkernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.kernelmethods.Forgetron;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.ForgetronBuilder;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestForgetron.
 */
public class TestForgetron {

	/** The dim. */
	int dim = (int) Math.pow(2, 18);

	/**
	 * Test.
	 */
	@Test
	public void test() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", (int) Math.pow(2, 18));
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		int folds = 2;
		for (int fold = 0; fold < folds; fold++) {

			Forgetron model = new Forgetron((int) Math.pow(2, 18), true);
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
	 * Test configure.
	 */
	@Test
	public void testConfigure() {
		ForgetronBuilder builder = new ForgetronBuilder(dim, false)
				.setDecay(3.).setWeight(1.2345).setPasses(23);

		Forgetron model = builder.build();
		assertEquals(23, model.getPasses());
	}

	/**
	 * Test nested configuration.
	 */
	@Test
	public void testNestedConfiguration() {
		ForgetronBuilder builder = new ForgetronBuilder(dim, false)
				.setDecay(3.).setWeight(1.2345).setPasses(23);

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
			Forgetron model = builder.build();
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
