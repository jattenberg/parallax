package com.dsi.parallax.ml.classifier.smoother;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.smoother.SmootherType;

public class TestIsotonicSmoother {

	int dim = (int) Math.pow(2, 14), min = 3;

	@Test
	public void testIsotonic() {
		SmootherTestUtils.testRegularization(SmootherType.ISOTONIC);
	}
}
