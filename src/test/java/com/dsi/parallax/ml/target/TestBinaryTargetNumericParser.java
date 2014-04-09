/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.target.BinaryTargetNumericParser;

public class TestBinaryTargetNumericParser {

	@Test
	public void testParsing() {
		String label = "x";
		BinaryTargetNumericParser parser = new BinaryTargetNumericParser();
		assertTrue(null == parser.parseTarget(label));
		label = null;
		assertTrue(null == parser.parseTarget(label));
		label = "1.5";
		assertTrue(null == parser.parseTarget(label));
		label = "1.0";
		assertTrue(null != parser.parseTarget(label));
		assertEquals(parser.parseTarget(label).getValue(), 1.0, 0.00000001);
		label = "0.5";
		assertTrue(null != parser.parseTarget(label));
		assertEquals(parser.parseTarget(label).getValue(), 0.5, 0.00000001);
		label = "0.0";
		assertTrue(null != parser.parseTarget(label));
		assertEquals(parser.parseTarget(label).getValue(), 0.0, 0.00000001);
	}

}
