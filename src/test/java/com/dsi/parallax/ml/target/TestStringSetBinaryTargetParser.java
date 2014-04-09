/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.ml.target.StringSetBinaryTargetParser;
import com.google.common.collect.Sets;

public class TestStringSetBinaryTargetParser {

	@Test
	public void testParsing() {
		StringSetBinaryTargetParser parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParser(Sets.newHashSet("x"), Sets.newHashSet("y"), true, false);
		String label = "x";
		BinaryClassificationTarget target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		label = "z";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParser(Sets.newHashSet("x"), Sets.newHashSet("y"), true, true);

		label = "z";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		
		parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParser(Sets.newHashSet("x"), Sets.newHashSet("y"), false, true);

		label = "z";
		target = parser.parseTarget(label);
		assertTrue(null == target);
		
		parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParser(null, Sets.newHashSet("y"), true, false);
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParser(Sets.newHashSet("x"), null, true, true);
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		
		
	}
	
	@Test
	public void testParsingNullMiss() {
		StringSetBinaryTargetParser parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParserNullMisses(Sets.newHashSet("x"), Sets.newHashSet("y"));
		String label = "x";
		BinaryClassificationTarget target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		label = "z";
		target = parser.parseTarget(label);
		assertTrue(null == target);
	}

	
	@Test
	public void testParsingLabeledMiss() {
		StringSetBinaryTargetParser parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParserPositiveMisses(Sets.newHashSet("x"), Sets.newHashSet("y"));
		String label = "x";
		BinaryClassificationTarget target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		label = "z";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		
		parser = StringSetBinaryTargetParser.buildStringSetBinaryTargetParserNegativeMisses(Sets.newHashSet("x"), Sets.newHashSet("y"));
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		label = "z";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
	}
	
	@Test
	public void testParsingPosNegSets() {
		StringSetBinaryTargetParser parser = StringSetBinaryTargetParser.buildStringSetPositiveExamplesNullMisses(Sets.newHashSet("x"));
		String label = "x";
		BinaryClassificationTarget target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null == target);
		
		parser = StringSetBinaryTargetParser.buildStringSetNegativeExamplesNullMisses(Sets.newHashSet("x"));
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null == target);
		
		parser = StringSetBinaryTargetParser.buildStringSetPositiveExamplesPosMiss(Sets.newHashSet("x"));
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		
		parser = StringSetBinaryTargetParser.buildStringSetNegativeExamplesPosMiss(Sets.newHashSet("x"));
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		
		
		parser = StringSetBinaryTargetParser.buildStringSetPositiveExamplesNegMiss(Sets.newHashSet("x"));
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 1.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
		parser = StringSetBinaryTargetParser.buildStringSetNegativeExamplesNegMiss(Sets.newHashSet("x"));
		label = "x";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		label = "y";
		target = parser.parseTarget(label);
		assertTrue(null != target);
		assertEquals(target.getValue(), 0.0, 0.0000001);
		
	}

}
