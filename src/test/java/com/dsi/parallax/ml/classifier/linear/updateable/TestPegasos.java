/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.linear.updateable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.ClassifierTestUtilities;
import com.dsi.parallax.ml.classifier.linear.updateable.Pegasos;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PegasosBuilder;
import com.dsi.parallax.ml.classifier.smoother.SmootherType;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestPegasos.
 */
public class TestPegasos {

	/** The dim. */
	int dim = (int) Math.pow(2, 14);

	@Test
	public void testCanClassify() {
		ClassifierTestUtilities.testCanClassify(new PegasosBuilder().setPasses(10));
		ClassifierTestUtilities.testCanClassifyText(new PegasosBuilder());
	}
	
	/**
	 * Test configuration.
	 */
	@Test
	public void testCanClassifyWithConfigTweaks() {
		PegasosBuilder builder = new PegasosBuilder().setPasses(10)
				.setWindowSize(15).setBias(true);
		ClassifierTestUtilities.testCanClassify(builder);
		ClassifierTestUtilities.testCanClassifyText(builder);
	}


	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		PegasosBuilder builder = new PegasosBuilder(dim, false)
				.setRegulizerType(SmootherType.UPDATEABLEPLATT).setDecay(3.)
				.setRegularizeIntercept(true).setRegularizationWeight(6)
				.setGaussianWeight(3.3).setLaplaceWeight(3.3)
				.setCauchyWeight(3.3).setSquaredWeight(3.3).setPasses(10)
				.setWindowSize(15);

		Pegasos model = builder.build();

		assertEquals(dim, model.getModelDimension());
		assertFalse(model.usesBiasTerm());
		assertEquals(15, model.getWindowSize());
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
		PegasosBuilder builder = new PegasosBuilder(dim, false);

		Pegasos base = builder.build();
		base.train(insts);

		double l1 = base.getVector().L1Norm();
		double l2 = base.getVector().L2Norm();

		Pegasos reg;
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

		builder.setLaplaceWeight(3);
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
		LinearUpdateableTestUtils.testTruncation(new PegasosBuilder());
	}
}
