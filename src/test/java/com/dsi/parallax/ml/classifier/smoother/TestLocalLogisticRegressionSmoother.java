package com.dsi.parallax.ml.classifier.smoother;

import org.junit.Test;
import org.junit.Ignore;

import com.dsi.parallax.ml.classifier.smoother.SmootherType;

public class TestLocalLogisticRegressionSmoother {

	@Test
	public void testRegularizer() {
		SmootherTestUtils
				.testRegularization(SmootherType.LOCALLR);
	}

    @Ignore
	@Test
	public void testPlattUpdateable() {
		SmootherTestUtils
				.testUpdateableRegularizer(SmootherType.LOCALLR);
	}

}
