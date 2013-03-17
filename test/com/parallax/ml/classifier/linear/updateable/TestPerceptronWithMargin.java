/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.linear.updateable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.classifier.ClassifierTestUtilities;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PerceptronWithMarginBuilder;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestPerceptronWithMargin.
 */
public class TestPerceptronWithMargin {

	/** The dim. */
	int dim = (int) Math.pow(2, 14);

	@Test
	public void testCanClassify() {
		ClassifierTestUtilities
				.testCanClassify(new PerceptronWithMarginBuilder());
		ClassifierTestUtilities
				.testCanClassifyText(new PerceptronWithMarginBuilder());

	}

	/**
	 * Test configuration.
	 */
	@Test
	public void testCanClassifyWithConfigTweaks() {
		PerceptronWithMarginBuilder builder = new PerceptronWithMarginBuilder()
				.setDecay(1.1)
				.setRegularizeIntercept(true)
				.setRegularizationWeight(0.1)
				.setGaussianWeight(3.3)
				.setLaplaceWeight(3.3)
				.setCauchyWeight(3.3)
				.setSquaredWeight(3.3)
				.setPasses(15)
				.setMargin(0)
				.setBias(true);

		ClassifierTestUtilities.testCanClassify(builder);
		ClassifierTestUtilities.testCanClassifyText(builder);
	}

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		PerceptronWithMarginBuilder builder = new PerceptronWithMarginBuilder(
				dim, false).setDecay(3.).setRegularizeIntercept(true)
				.setRegularizationWeight(6).setGaussianWeight(3.3)
				.setLaplaceWeight(3.3).setCauchyWeight(3.3)
				.setSquaredWeight(3.3).setPasses(10).setMargin(.12345);

		PerceptronWithMargin model = builder.build();

		assertEquals(dim, model.getModelDimension());
		assertFalse(model.usesBiasTerm());
		assertEquals(.12345, model.getMargin(), 0);
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

	@Test
	public void testRegularization() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", dim);
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();
		PerceptronWithMarginBuilder builder = new PerceptronWithMarginBuilder(
				dim, false);

		PerceptronWithMargin base = builder.build();
		base.train(insts);

		double l1 = base.getVector().L1Norm();
		double l2 = base.getVector().L2Norm();

		PerceptronWithMargin reg;
		builder.setCauchyWeight(3);
		reg = builder.build();
		reg.train(insts);
		assertTrue(reg.getVector().L1Norm() < l1);
		assertTrue(reg.getVector().L2Norm() < l2);

		builder.setCauchyWeight(0);
		builder.setGaussianWeight(3);
		reg = builder.build();
		reg.train(insts);
		assertTrue(reg.getVector().L1Norm() < l1);
		assertTrue(reg.getVector().L2Norm() < l2);

		builder.setGaussianWeight(0);

		builder.setLaplaceWeight(3.);
		reg = builder.build();
		reg.train(insts);

		assertTrue(reg.getVector().L1Norm() < l1);
		assertTrue(reg.getVector().L2Norm() < l2);
		builder.setLaplaceWeight(0);

		builder.setSquaredWeight(3);
		reg = builder.build();
		reg.train(insts);

		assertTrue(reg.getVector().L1Norm() < l1);
		assertTrue(reg.getVector().L2Norm() < l2);
	}

	@Test
	public void testTruncation() {
		LinearUpdateableTestUtils
				.testTruncation(new PerceptronWithMarginBuilder());
	}
}
