/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.trees;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.trees.MaximumDepthTerminator;

/**
 * The Class TestMaxDepthTerminator.
 */
public class TestMaxDepthTerminator {

	/**
	 * Test.
	 */
	@Test
	public void test() {
		MaximumDepthTerminator<BinaryClassificationTarget> term = new MaximumDepthTerminator<BinaryClassificationTarget>(
				50);
		for (int i = 0; i < 100; i++)
			assertEquals(term.terminate(null, i), i >= 50);

	}

}
