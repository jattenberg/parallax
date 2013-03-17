/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.classifier.linear.updateable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.classifier.ClassifierTestUtilities;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.AROWClassifierBuilder;
import com.parallax.ml.classifier.smoother.SmootherType;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.utils.ScienceReader;
import com.parallax.optimization.regularization.LinearCoefficientLossType;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.regularization.TruncationType;

/**
 * The Class TestAROWClassifier.
 */
public class TestAROWClassifier {

	@Test
	public void testCanClassify() {
		ClassifierTestUtilities.testCanClassify(new AROWClassifierBuilder()
				.setR(1).setPasses(5));
		ClassifierTestUtilities.testCanClassifyText(new AROWClassifierBuilder()
				.setR(1).setPasses(5));
	}

	@Test
	public void testClassifiesWithConfigTweaks() {
		AROWClassifierBuilder builder = new AROWClassifierBuilder().setR(190.6)
				.setPasses(5).setRegulizerType(SmootherType.UPDATEABLEPLATT)
				.setBias(true);

		ClassifierTestUtilities.testCanClassify(builder);
		ClassifierTestUtilities.testCanClassifyText(builder);
	}

	/** The regularization. */
	double gauss = .333, cauchy = .666, laplace = .999, squared = 1.333,
			regularization = 1.666;

	/** The rtype. */
	SmootherType rtype = SmootherType.UPDATEABLEPLATT;

	/** The t builder. */
	TruncationConfigurableBuilder tBuilder = new TruncationConfigurableBuilder()
			.setTruncationType(TruncationType.TRUNCATING);

	/** The intercept. */
	boolean intercept = false;

	/**
	 * Test setters.
	 */
	@Test
	public void testSetters() {
		AROWClassifier model = new AROWClassifier(ScienceReader.DIMENSION, true);
		model.setGaussianRegularizationWeight(.5)
				.setCauchyRegularizationWeight(.5)
				.setLaplaceRegularizationWeight(.5)
				.setSquaredRegularizationWeight(.5)
				.setR(33)
				.setTruncationBuilder(
						new TruncationConfigurableBuilder()
								.setTruncationType(TruncationType.TRUNCATING))
				.setRegularizeIntercept(false).setRegularizationWeight(0.33333)
				.setSmoothertype(SmootherType.UPDATEABLEPLATT).initialize();
	}

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		AROWClassifierBuilder builder = new AROWClassifierBuilder(
				ScienceReader.DIMENSION, false);
		builder.setR(190.6).setRegulizerType(SmootherType.UPDATEABLEPLATT)
				.setDecay(3.).setRegularizeIntercept(true)
				.setRegularizationWeight(6).setGaussianWeight(3.3)
				.setLaplaceWeight(3.3).setCauchyWeight(3.3)
				.setSquaredWeight(3.3).setPasses(10);

		AROWClassifier model = builder.build();
		assertEquals(model.getModelDimension(), ScienceReader.DIMENSION);
		assertTrue(!model.usesBiasTerm());
		assertEquals(model.getR(), 190.6, 0);

		assertEquals(ScienceReader.DIMENSION, model.getModelDimension());
		assertFalse(model.usesBiasTerm());
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
		BinaryClassificationInstances insts = ScienceReader.readScience();
		AROWClassifierBuilder builder = new AROWClassifierBuilder(
				ScienceReader.DIMENSION, false);

		AROWClassifier base = builder.build();
		base.train(insts);

		double l1 = base.getVector().L1Norm();
		double l2 = base.getVector().L2Norm();

		AROWClassifier reg;
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

		builder.setLaplaceWeight(25.);
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
		LinearUpdateableTestUtils.testTruncation(new AROWClassifierBuilder());
	}
}
