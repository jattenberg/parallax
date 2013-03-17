package com.parallax.ml.classifier.smoother;

import org.junit.Test;

import com.parallax.ml.classifier.smoother.SmootherType;

public class TestBinningSmoother {

	int dim = (int) Math.pow(2, 14), min = 3;

	@Test
	public void testRegularizer() {
		SmootherTestUtils.testRegularization(SmootherType.BINNING);
	}
}
