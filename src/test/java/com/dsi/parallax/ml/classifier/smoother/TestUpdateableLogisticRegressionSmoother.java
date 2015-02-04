package com.dsi.parallax.ml.classifier.smoother;

import org.junit.Test;
import org.junit.Ignore;

import com.dsi.parallax.ml.classifier.smoother.SmootherType;

public class TestUpdateableLogisticRegressionSmoother {

	@Test
	public void testRegularizer() {
		SmootherTestUtils
				.testRegularization(SmootherType.UPDATEABLEPLATT);
	}

	@Test
    @Ignore
	public void testPlattUpdateable() {
		SmootherTestUtils
				.testUpdateableRegularizer(SmootherType.UPDATEABLEPLATT);
	}
}
