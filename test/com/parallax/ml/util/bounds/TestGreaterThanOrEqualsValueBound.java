package com.parallax.ml.util.bounds;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class TestGreaterThanOrEqualsValueBound {

	@Test
	public void test() {
		GreaterThanOrEqualsValueBound bound = new GreaterThanOrEqualsValueBound(1);
		assertTrue(bound.satisfiesBound(2));
		assertTrue(bound.satisfiesBound(1));
		assertFalse(bound.satisfiesBound(0));
	}

}
