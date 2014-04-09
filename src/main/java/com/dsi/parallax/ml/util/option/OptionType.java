/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

/**
 * OptionType is enum class defines detail Option type
 *  BOOLEAN - BooleanOption
 *  INTEGER - IntegerOption
 *  FLOAT - FloatOption
 *  ENUM - EnumOption - an option representing an enum class
 *  CONFIGURABLE - ConfigurableOption - an option that is recursively configurable
 */
public enum OptionType
{
	BOOLEAN, INTEGER, FLOAT, ENUM, CONFIGURABLE, NESTEDCONFIGURABLE;
}
