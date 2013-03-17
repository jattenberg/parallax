/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions.kde;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.util.MLUtils;

/**
 * The Class TestUnivariateKernelDensityEstimator.
 */
public class TestUnivariateKernelDensityEstimator {

	/**
	 * Test basic kde.
	 */
	@Test
	public void testBasicKDE() {
		for (KDEKernel kernel : KDEKernel.values()) {
			UnivariateKernelDensityEstimator kde = new UnivariateKernelDensityEstimator(
					kernel);
			for (int i = 0; i < 5000; i++)
				kde.observe(i);
			for (int i = 0; i < 5000; i++) {
				double d = kde.probability(i);
				assertTrue(d >= 0 && d <= 1);
			}
			for (int j = 0; j < 5; j++) {
				kde = new UnivariateKernelDensityEstimator(kernel);
				for (int i = 0; i < 50000; i++)
					kde.observe(MLUtils.GENERATOR.nextGaussian() * 250);
				for (int i = 0; i < 50; i++) {
					double d = kde.probability(i * 10);
					assertTrue(d >= 0 && d <= 1);
				}
			}
		}
	}

}
