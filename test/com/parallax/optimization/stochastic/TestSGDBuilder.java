/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.stochastic;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TestSGDBuilder {

	@Test
	public void test() {
		SGDBuilder builder = new SGDBuilder(10, true);
		StochasaticGradientDescent sgd = builder.build();
				
		assertEquals(sgd.getDimension(), 11);
		assertTrue(sgd.isBias());
	}

}
