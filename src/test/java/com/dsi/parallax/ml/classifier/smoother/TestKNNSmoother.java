package com.dsi.parallax.ml.classifier.smoother;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.smoother.SmootherType;

public class TestKNNSmoother {

	@Test
	public void testRegularizer() {
		SmootherTestUtils
				.testRegularization(SmootherType.KNN);
	}

	@Test
	public void testPlattUpdateable() {
		SmootherTestUtils
				.testUpdateableRegularizer(SmootherType.KNN);
	}

}
 