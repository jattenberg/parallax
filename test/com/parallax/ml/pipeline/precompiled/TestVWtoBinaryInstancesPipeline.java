/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.pipeline.precompiled;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.pipeline.precompiled.VWtoBinaryInstancesPipeline;

/**
 * The Class TestVWtoBinaryInstancesPipeline.
 */
public class TestVWtoBinaryInstancesPipeline {

	/**
	 * Test.
	 */
	@Test
	public void test() {
		VWtoBinaryInstancesPipeline pipe = new VWtoBinaryInstancesPipeline(
				"data/science.small.vw", (int) Math.pow(2, 18));
		assertTrue(pipe.hasNext());
		BinaryClassificationInstances insts = pipe.next();
		assertTrue(!pipe.hasNext());
		assertEquals(insts.size(), 200);

		assertEquals(100, insts.getNumNeg());
		assertEquals(100, insts.getNumPos());
	}

}
