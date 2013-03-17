/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.util.bounds.GreaterThanValueBound;
import com.parallax.ml.util.bounds.LessThanValueBound;

public class TestIntegerOption {
	@Test(expected = IllegalArgumentException.class)
	public void testCheckCurrent() {
		IntegerOption integerOption = getInstance();
		// test for positive
		assertTrue(integerOption.checkCurrent(6) == 6);
		// test for negative
		integerOption.checkCurrent(0);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testMixedBounds() {
		String shortName = "test_shortName";
		String longName = "test_longName";
		String desc = "test_desc";
		int min = 10;
		int max = 1;
		int defaultv = 5;
		boolean optimizable = false;
		new IntegerOption(shortName, longName, desc, defaultv,
				optimizable, new GreaterThanValueBound(min),
				new LessThanValueBound(max));
	}

	@Test
	public void testCopyOption() {
		IntegerOption integerOption = getInstance();
		IntegerOption copyOptionObject = (IntegerOption) integerOption
				.copyOption();
		assertTrue(integerOption.getLowerBound().getBoundValue() == copyOptionObject
				.getLowerBound().getBoundValue());
		assertTrue(integerOption.getUpperBound().getBoundValue() == copyOptionObject
				.getUpperBound().getBoundValue());
		assertTrue(integerOption.getDEFAULT() == copyOptionObject.getDEFAULT());
		assertEquals(integerOption.getDescription(),
				copyOptionObject.getDescription());
		assertEquals(integerOption.getLongName(),
				copyOptionObject.getLongName());
		assertEquals(integerOption.getShortName(),
				copyOptionObject.getShortName());
	}

	private IntegerOption getInstance() {
		String shortName = "test_shortName";
		String longName = "test_longName";
		String desc = "test_desc";
		int min = 1;
		int max = 10;
		int defaultv = 5;
		boolean optimizable = false;
		return new IntegerOption(shortName, longName, desc, defaultv,
				optimizable, new GreaterThanValueBound(min),
				new LessThanValueBound(max));
	}
}
