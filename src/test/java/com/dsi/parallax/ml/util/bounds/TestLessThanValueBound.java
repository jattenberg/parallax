package com.dsi.parallax.ml.util.bounds;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.util.bounds.LessThanValueBound;

public class TestLessThanValueBound {

	@Test
	public void test() {
		LessThanValueBound bound = new LessThanValueBound(1);
		assertFalse(bound.satisfiesBound(2));
		assertFalse(bound.satisfiesBound(1));
		assertTrue(bound.satisfiesBound(0));
	}

}
