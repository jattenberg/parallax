
/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.linear.updateable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.classifier.ClassifierTestUtilities;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PassiveAggressiveBuilder;
import com.parallax.ml.classifier.smoother.SmootherType;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.util.option.Configuration;
import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestPassiveAggressive.
 */
public class TestPassiveAggressive {

	/** The dim. */
	int dim = (int) Math.pow(2, 14);

	@Test
	public void testCanClassify() {
		ClassifierTestUtilities.testCanClassify(new PassiveAggressiveBuilder());
		ClassifierTestUtilities
				.testCanClassifyText(new PassiveAggressiveBuilder());
	}

	/**
	 * Test configuration.
	 */
	@Test
	public void testCanClassifyWithConfigTweaks() {
		PassiveAggressiveBuilder builder = new PassiveAggressiveBuilder()
				.setAggressiveness(17.3)
				.setRegulizerType(SmootherType.ISOTONIC)
				.setCrossvalidateSmootherTraining(2)
				.setRegularizeIntercept(false)
				.setRegularizationWeight(0.1)
				.setGaussianWeight(3.3)
				.setLaplaceWeight(3.3)
				.setCauchyWeight(3.3)
				.setSquaredWeight(3.3)
				.setPasses(5)
				.setBias(true);
		ClassifierTestUtilities.testCanClassify(builder);
		ClassifierTestUtilities.testCanClassifyText(builder);
	}

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		PassiveAggressiveBuilder builder = new PassiveAggressiveBuilder(dim,
				false).setAggressiveness(33)
				.setRegulizerType(SmootherType.UPDATEABLEPLATT).setDecay(3.)
				.setRegularizeIntercept(true).setRegularizationWeight(6)
				.setGaussianWeight(3.3).setLaplaceWeight(3.3)
				.setCauchyWeight(3.3).setSquaredWeight(3.3).setPasses(10);

		PassiveAggressive model = builder.build();

		assertEquals(dim, model.getModelDimension());
		assertFalse(model.usesBiasTerm());
		assertEquals(33, model.getAggressiveness(), 0);
		assertTrue(model.isRegularizeIntercept());
		assertEquals(10, model.getPasses(), 0);
		assertEquals(3.3, model.getGaussianRegularizationWeight(), 0);
		assertEquals(3.3, model.getLaplaceRegularizationWeight(), 0);
		assertEquals(3.3, model.getCauchyRegularizationWeight(), 0);
		assertEquals(3.3, model.getSquaredRegularizationWeight(), 0);
		assertEquals(
				0,
				model.getRegularizationTypeWeight(LinearCoefficientLossType.UNIFORM),
				0);

	}

	/**
	 * Test regularization.
	 */
	@Test
	public void testRegularization() {
		PassiveAggressiveBuilder builder = new PassiveAggressiveBuilder(dim,
				false);

		PassiveAggressive model = builder.build();

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();
		model.train(insts);

		double beforeNorm = model.getVector().L2Norm();
		double beforeOneNorm = model.getVector().L1Norm();
		String[] regNames = new String[] { "GR", "LR", "CR", "SR" };
		for (String reg : regNames) {
			Configuration<PassiveAggressiveBuilder> conf = builder
					.getConfiguration();
			conf.addIntegerValueOnShortName("d", dim);
			conf.addFloatValueOnShortName(reg, 5);
			PassiveAggressiveBuilder builder2 = new PassiveAggressiveBuilder(
					conf);
			model = builder2.build();
			model.train(insts);
			assertTrue(beforeOneNorm > model.getVector().L1Norm());
			assertTrue(beforeNorm > model.getVector().L2Norm());
		}
	}

	@Test
	public void testTruncation() {
		LinearUpdateableTestUtils
				.testTruncation(new PassiveAggressiveBuilder());
	}
}
