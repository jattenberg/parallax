package com.parallax.ml.util.bounds;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TestGreaterThanValueBound {

	@Test
	public void test() {
		GreaterThanValueBound bound = new GreaterThanValueBound(1);
		assertTrue(bound.satisfiesBound(2));
		assertFalse(bound.satisfiesBound(1));
		assertFalse(bound.satisfiesBound(0));
	}

}
