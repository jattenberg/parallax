/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanValueBound;
import com.dsi.parallax.ml.util.option.FloatOption;

public class TestFloatOption {

	@Test(expected = IllegalArgumentException.class)
	public void testCheckCurrent() {
		FloatOption floatOption = getInstance();
		// test for positive
		assertTrue(floatOption.checkCurrent(6) == 6);
		// test for negative
		floatOption.checkCurrent(0);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testMixedBounds() {
		String shortName = "test_shortName";
		String longName = "test_longName";
		String desc = "test_desc";
		double min = 10;
		double max = 1;
		double defaultv = 5;
		boolean optimizable = false;
		new FloatOption(shortName, longName, desc, defaultv,
				optimizable, new GreaterThanValueBound(min),
				new LessThanValueBound(max));
	}

	@Test
	public void testCopyOption() {
		FloatOption floatOption = getInstance();
		FloatOption copyOptionObject = (FloatOption) floatOption.copyOption();
		assertTrue(floatOption.getLowerBound().getBoundValue() == copyOptionObject
				.getLowerBound().getBoundValue());
		assertTrue(floatOption.getUpperBound().getBoundValue() == copyOptionObject
				.getUpperBound().getBoundValue());
		assertTrue(floatOption.getDEFAULT() == copyOptionObject.getDEFAULT());
		assertEquals(floatOption.getDescription(),
				copyOptionObject.getDescription());
		assertEquals(floatOption.getLongName(), copyOptionObject.getLongName());
		assertEquals(floatOption.getShortName(),
				copyOptionObject.getShortName());
	}

	private FloatOption getInstance() {
		String shortName = "test_shortName";
		String longName = "test_longName";
		String desc = "test_desc";
		double min = 1;
		double max = 10;
		double defaultv = 5;
		boolean optimizable = false;
		return new FloatOption(shortName, longName, desc, defaultv,
				optimizable, new GreaterThanValueBound(min),
				new LessThanValueBound(max));
	}
}
