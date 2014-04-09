/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.mercerkernels;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.kernelmethods.KernelSGD;
import com.dsi.parallax.ml.classifier.kernelmethods.KernelMethodBuilder.KernelSGDBuilder;
import com.dsi.parallax.ml.evaluation.LossGradientType;
import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;


/**
 * The Class TestKernelSGD.
 */
public class TestKernelSGD {

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

			KernelSGD model = new KernelSGD(dim, true);
			OnlineEvaluation eval = new OnlineEvaluation();

			model.train(insts.getTraining(fold, folds));
			for (BinaryClassificationInstance x : insts.getTesting(fold, folds)) {
				double pred = model.predict(x).getValue();
				double label = x.getLabel().getValue();
				eval.add(label, pred);
			}

			assertTrue(eval.computeAccuracy() > 0.5);
		}
	}

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		KernelSGDBuilder builder = new KernelSGDBuilder(dim, false)
				.setDecay(3.).setMargin(.123).setEta(.12345)

				.setLossGradientType(LossGradientType.LOGLOSS).setLambda(.3)
				.setWeight(1.2345).setPasses(23);

		KernelSGD model = builder.build();
		assertEquals(23, model.getPasses());
		assertEquals(.12345, model.getEta(), 0);
		assertEquals(.3, model.getLambda(), 0);
		assertEquals(LossGradientType.LOGLOSS, model.getLossType());

	}

	/**
	 * Test configuration.
	 */
	@Test
	public void testConfiguration() {
		KernelSGDBuilder builder = new KernelSGDBuilder(dim, false)
				.setDecay(3.).setEta(.12345)
				.setLossGradientType(LossGradientType.LOGLOSS).setLambda(.3)
				.setWeight(1.2345).setPasses(23);

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();

		int folds = 3;
		for (int fold = 0; fold < folds; fold++) {
			BinaryClassificationInstances train = insts
					.getTraining(fold, folds);
			BinaryClassificationInstances test = insts.getTesting(fold, folds);
			assertTrue(train.getNumNeg() > 0);
			assertTrue(test.getNumNeg() > 0);
			assertTrue(train.getNumPos() > 0);
			assertTrue(train.getNumPos() > 0);
			KernelSGD model = builder.build();

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
