/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.parallax.ml.target.BinaryClassificationTarget;

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
