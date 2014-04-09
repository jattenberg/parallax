package com.dsi.parallax.ml.util.bounds;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;

public class TestLessThanOrEqualsValueBound {

	@Test
	public void test() {
		LessThanOrEqualsValueBound bound = new LessThanOrEqualsValueBound(1);
		assertFalse(bound.satisfiesBound(2));
		assertTrue(bound.satisfiesBound(1));
		assertTrue(bound.satisfiesBound(0));
	}

}
