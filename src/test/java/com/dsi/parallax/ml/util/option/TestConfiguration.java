/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.lang.reflect.InvocationTargetException;

import org.apache.commons.cli.ParseException;
import org.junit.Test;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanValueBound;
import com.dsi.parallax.ml.util.option.BooleanOption;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.EnumOption;
import com.dsi.parallax.ml.util.option.FloatOption;
import com.dsi.parallax.ml.util.option.IntegerOption;
import com.dsi.parallax.ml.util.option.Option;

public class TestConfiguration {
	@Test
	public void testAddOption() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.getCliOpts().getOptions().size() == 4);
	}

	@Test
	public void testAddFloatValueOnShortName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addFloatValueOnShortName(
				"test_shortName_Float", 8);
		assertTrue(configuration
				.floatOptionFromShortName("test_shortName_Float") == 8);
	}

	@Test
	public void testAddFloatValueOnLongName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addFloatValueOnLongName(
				"test_longName_Float", 7);
		assertTrue(configuration.floatFromLongName("test_longName_Float") == 7);
	}

	@Test
	public void testAddIntegerValueOnShortName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addIntegerValueOnShortName(
				"test_shortName_Integer", 8);
		assertTrue(configuration
				.integerOptionFromShortName("test_shortName_Integer") == 8);
	}

	@Test
	public void testAddIntegerValueOnLongName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addIntegerValueOnLongName(
				"test_longName_Integer", 7);
		assertTrue(configuration.integerFromLongName("test_longName_Integer") == 7);
	}

	@Test
	public void testAddBooleanValueOnShortName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addBooleanValueOnShortName(
				"test_shortName_Boolean", false);
		assertFalse(configuration
				.booleanOptionFromShortName("test_shortName_Boolean"));
	}

	@Test
	public void testAddBooleanValueOnLongName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addBooleanValueOnLongName(
				"test_longName_Boolean", false);
		assertFalse(configuration.booleanFromLongName("test_longName_Boolean"));
	}

	@Test
	public void testAddEnumValueOnShortName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addEnumValueOnShortName(
				"test_shortName_Enum", EnumTest.TEST2);
		assertTrue(configuration.enumFromShortName("test_shortName_Enum")
				.equals(EnumTest.TEST2));
	}

	@Test
	public void testAddEnumValueOnLongName() {
		Configuration<?> configuration = getInstance();
		configuration = configuration.addEnumValueOnLongName(
				"test_longName_Enum", EnumTest.TEST2);
		assertTrue(configuration.enumFromLongName("test_longName_Enum").equals(
				EnumTest.TEST2));
	}

	@Test
	public void testEnumFromShortName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.enumFromShortName("test_shortName_Enum") == EnumTest.TEST1);
	}

	@Test
	public void testEnumFromLongName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.enumFromLongName("test_longName_Enum") == EnumTest.TEST1);
	}

	@Test
	public void testIntegerOptionFromShortName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration
				.integerOptionFromShortName("test_shortName_Integer") == 5);
	}

	@Test
	public void testIntegerFromLongName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.integerFromLongName("test_longName_Integer") == 5);
	}

	@Test
	public void testFloatOptionFromShortName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration
				.floatOptionFromShortName("test_shortName_Float") == 5);
	}

	@Test
	public void testFloatFromLongName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.floatFromLongName("test_longName_Float") == 5);
	}

	@Test
	public void testBooleanOptionFromShortName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration
				.booleanOptionFromShortName("test_shortName_Boolean"));
	}

	@Test
	public void testBooleanFromLongName() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.booleanFromLongName("test_longName_Boolean"));
	}

	@Test()
	public void testContainsShortKey() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.containsShortKey("test_shortName_Float"));
		assertTrue(configuration.containsShortKey("test_shortName_Integer"));
		assertTrue(configuration.containsShortKey("test_shortName_Boolean"));
		assertTrue(configuration.containsShortKey("test_shortName_Enum"));
	}

	@Test
	public void testContainsLongKey() {
		Configuration<?> configuration = getInstance();
		assertTrue(configuration.containsLongKey("test_longName_Float"));
		assertTrue(configuration.containsLongKey("test_longName_Integer"));
		assertTrue(configuration.containsLongKey("test_longName_Boolean"));
		assertTrue(configuration.containsLongKey("test_longName_Enum"));
	}

	@Test
	public void testCopyConfiguration() {
		Configuration<?> configuration = getInstance();
		Configuration<?> newConfiguration = configuration.copyConfiguration();
		assertEquals(configuration.getCliOpts().getOptions().size(),
				newConfiguration.getCliOpts().getOptions().size());
	}

	private Option getFloatOption() {
		String shortName = "test_shortName_Float";
		String longName = "test_longName_Float";
		String desc = "test_desc";
		double min = 1;
		double max = 10;
		double defaultv = 5;
		boolean optimizable = false;

		return new FloatOption(shortName, longName, desc, defaultv,
				optimizable, new GreaterThanValueBound(min),
				new LessThanValueBound(max));
	}

	private Option getIntegerOption() {
		String shortName = "test_shortName_Integer";
		String longName = "test_longName_Integer";
		String desc = "test_desc";
		int min = 1;
		int max = 10;
		int defaultv = 5;
		boolean optimizable = false;

		return new IntegerOption(shortName, longName, desc, defaultv,
				optimizable, new GreaterThanValueBound(min),
				new LessThanValueBound(max));
	}

	private Option getBooleanOption() {
		String shortName = "test_shortName_Boolean";
		String longName = "test_longName_Boolean";
		String desc = "test_desc";
		boolean defaultv = true;
		boolean optimizable = false;

		return new BooleanOption(shortName, longName, desc, defaultv,
				optimizable);
	}

	@SuppressWarnings({ "rawtypes", "unchecked" })
	private Option getEnumOption() {
		String shortName = "test_shortName_Enum";
		String longName = "test_longName_Enum";
		String desc = "test_desc";
		boolean optimizable = false;

		return new EnumOption(shortName, longName, optimizable, desc,
				EnumTest.class, EnumTest.TEST1);
	}

	private Configuration<?> getInstance() {
		@SuppressWarnings("rawtypes")
		Configuration<?> configuration = new Configuration();
		configuration.addOption(getFloatOption());
		configuration.addOption(getIntegerOption());
		configuration.addOption(getBooleanOption());
		configuration.addOption(getEnumOption());
		return configuration;
	}

	public enum EnumTest {
		TEST1, TEST2
	}

	@Test
	public void testOptionValuesFromArgs() throws InvocationTargetException,
			NoSuchMethodException, ParseException, IllegalAccessException {
		Configuration<?> configuration = getInstance();
		String[] args = new String[] { "-test_shortName_Float", "5.3",
				"-test_shortName_Integer", "6", "-test_shortName_Boolean",
				"false", "-test_shortName_Enum", "TEST2" };
		configuration.optionValuesFromArgs(args);
		assertTrue(configuration
				.floatOptionFromShortName("test_shortName_Float") == 5.3);
		assertTrue(configuration
				.integerOptionFromShortName("test_shortName_Integer") == 6);
		assertFalse(configuration
				.booleanOptionFromShortName("test_shortName_Boolean"));
		assertEquals(configuration.enumFromShortName("test_shortName_Enum"),
				EnumTest.TEST2);
	}

	@Test
	public void testGetArgumentsFromOpts() {
		Configuration<?> configuration = getInstance();
		String[] array = configuration.getArgumentsFromOpts();
		assertTrue(array.length == 6);
	}
}
