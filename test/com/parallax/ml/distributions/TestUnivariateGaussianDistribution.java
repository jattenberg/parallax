/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.distributions;

import static org.junit.Assert.assertEquals;

import org.apache.commons.math.distribution.NormalDistribution;
import org.apache.commons.math.distribution.NormalDistributionImpl;
import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import org.junit.Test;

/**
 * The Class TestUnivariateGaussianDistribution.
 */
public class TestUnivariateGaussianDistribution {

	/**
	 * Test.
	 */
	@Test
	public void test() {
		UnivariateGaussianDistribution gd = new UnivariateGaussianDistribution();
		for (int i = 0; i < 5; i++)
			assertEquals(gd.probability(i), 0, 0.00001);
		DescriptiveStatistics ds = new DescriptiveStatistics();
		for (int i = 0; i < 5; i++) {
			gd.observe(i);
			ds.addValue(i);
			assertEquals(gd.getMean(), ds.getMean(), 0.001);
			assertEquals(gd.getVariance(), ds.getVariance(), 0.001);

		}
		NormalDistribution nd = new NormalDistributionImpl(ds.getMean(),
				ds.getStandardDeviation());
		for (int i = 0; i < 5; i++) {
			assertEquals(gd.probability(i), nd.density((double) i), 0.0001);
		}

	}

}
