package com.dsi.parallax.ml.classifier.smoother;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.smoother.SmootherType;

/**
 * The Class TestIsotonicRegression.
 */
public class TestLogisticRegressionSmoother {

	@Test
	public void testRegularizer() {
		SmootherTestUtils.testRegularization(SmootherType.PLATT);
	}
}
