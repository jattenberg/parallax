package com.parallax.ml.classifier.smoother;

import org.junit.Test;

import com.parallax.ml.classifier.smoother.SmootherType;

public class TestLocalLogisticRegressionSmoother {

	@Test
	public void testRegularizer() {
		SmootherTestUtils
				.testRegularization(SmootherType.LOCALLR);
	}

	@Test
	public void testPlattUpdateable() {
		SmootherTestUtils
				.testUpdateableRegularizer(SmootherType.LOCALLR);
	}

}
