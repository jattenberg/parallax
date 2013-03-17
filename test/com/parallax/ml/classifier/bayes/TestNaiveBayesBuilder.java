package com.parallax.ml.classifier.bayes;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import org.junit.Test;

import com.parallax.ml.classifier.smoother.SmootherType;
import com.parallax.ml.distributions.DistributionConfigurableBuilder;
import com.parallax.ml.distributions.DistributionType;
import com.parallax.ml.distributions.kde.KDEConfigurableBuilder;
import com.parallax.ml.distributions.kde.KDEKernel;

/**
 * The Class TestNaiveBayesBuilder.
 */
public class TestNaiveBayesBuilder {

	/** The dimension. */
	int dimension = (int) Math.pow(2, 18);

	/**
	 * Test options.
	 */
	@Test
	public void testOptions() {
		NaiveBayesBuilder builder = new NaiveBayesBuilder(dimension, false)
				.setDecay(1.2345).setDocumentLengthNormalization(456)
				.setPasses(23)
				.setRegulizerType(SmootherType.UPDATEABLEPLATT)
				.setWeight(1.2345);
		NaiveBayes model = builder.build();

		assertEquals(dimension, model.getModelDimension());
		assertFalse(model.usesBiasTerm());
		assertEquals(456, model.getDocLengthNormalization(), 0);
		assertEquals(1, model.getPasses());
		assertEquals(SmootherType.UPDATEABLEPLATT, model.getSmoothertype());
	}

	/**
	 * Test distribution builder options.
	 */
	@Test
	public void testDistributionBuilderOptions() {
		NaiveBayesBuilder builder = new NaiveBayesBuilder(dimension, false);

		DistributionConfigurableBuilder distBuilder = new DistributionConfigurableBuilder()
				.setAlpha(.123).setBins(5)
				.setDistributionType(DistributionType.BERNOULLI);
		builder.setDistributionBuilder(distBuilder);
		NaiveBayes model = builder.build();

		DistributionConfigurableBuilder distBuilder2 = model
				.getDistributionBuilder();

		assertEquals(.123, distBuilder2.getAlpha(), 0);
		assertEquals(5, distBuilder2.getBins());
		assertEquals(DistributionType.BERNOULLI,
				distBuilder2.getDistributionType());
	}

	/**
	 * Test kernel distribution distribution builder options.
	 */
	@Test
	public void testKernelDistributionDistributionBuilderOptions() {
		NaiveBayesBuilder builder = new NaiveBayesBuilder(dimension, false);

		KDEConfigurableBuilder kernelBuilder = new KDEConfigurableBuilder();
		kernelBuilder.setBandWidth(5).setDistanceDamping(2)
				.setKernelType(KDEKernel.TRIANGULAR);

		DistributionConfigurableBuilder distBuilder = new DistributionConfigurableBuilder()
				.setAlpha(.123).setBins(5)
				.setDistributionType(DistributionType.KDE)
				.setKDEBuilder(kernelBuilder);
		builder.setDistributionBuilder(distBuilder);
		NaiveBayes model = builder.build();

		DistributionConfigurableBuilder distBuilder2 = model
				.getDistributionBuilder();
		KDEConfigurableBuilder kdernelBuilder2 = distBuilder2.getKdeBuilder();

		assertEquals(5, kdernelBuilder2.getBandwidth());
		assertEquals(2, kdernelBuilder2.getDistanceDamping(), 0);
		assertEquals(KDEKernel.TRIANGULAR, kdernelBuilder2.getKernel());
	}

}
