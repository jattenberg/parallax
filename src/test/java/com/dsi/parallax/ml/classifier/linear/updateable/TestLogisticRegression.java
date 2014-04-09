/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.ClassifierTestUtilities;
import com.dsi.parallax.ml.classifier.linear.updateable.LogisticRegression;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.LogisticRegressionBuilder;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestLogisticRegression.
 */
public class TestLogisticRegression {

	/** The dim. */
	int dim = (int) Math.pow(2, 14);

	@Test
	public void testCanClassify() {
		ClassifierTestUtilities
				.testCanClassify(new LogisticRegressionBuilder());
		ClassifierTestUtilities
				.testCanClassifyText(new LogisticRegressionBuilder());
	}

	/**
	 * Test configuration.
	 */
	@Test
	public void testCanClassifyWithConfigTweaks() {
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder()
				.setDecay(3.).setGamma(.333).setRegularizeIntercept(true)
				.setRegularizationWeight(6).setShift(true)
				.setGaussianWeight(3.3).setPasses(10).setBias(true);
		ClassifierTestUtilities.testCanClassify(builder);
		ClassifierTestUtilities.testCanClassifyText(builder);
	}

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(dim,
				false).setDecay(3.).setGamma(.333).setRegularizeIntercept(true)
				.setRegularizationWeight(6).setShift(true)
				.setGaussianWeight(3.3).setLaplaceWeight(3.3)
				.setCauchyWeight(3.3).setSquaredWeight(3.3).setPasses(10);

		LogisticRegression model = builder.build();
		assertEquals(dim, model.getModelDimension());
		assertFalse(model.usesBiasTerm());
		assertTrue(model.isShift());
		assertEquals(.333, model.getGamma(), 0);
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
		LogisticRegressionBuilder builder = new LogisticRegressionBuilder(dim,
				false);
		LogisticRegression model = builder.build();

		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();
		model.train(insts);

		double beforeNorm = model.getVector().L2Norm();
		String[] regNames = new String[] { "GR", "LR", "CR", "SR" };
		for (String reg : regNames) {
			Configuration<LogisticRegressionBuilder> conf = builder
					.getConfiguration();
			conf.addIntegerValueOnShortName("d", dim);
			conf.addFloatValueOnShortName(reg, 5.0);
			LogisticRegressionBuilder builder2 = new LogisticRegressionBuilder(
					conf);
			model = builder2.build();
			model.train(insts);
			assertTrue(beforeNorm > model.getVector().L2Norm());
		}
	}

	@Test
	public void testTruncation() {
		LinearUpdateableTestUtils
				.testTruncation(new LogisticRegressionBuilder());
	}
}
