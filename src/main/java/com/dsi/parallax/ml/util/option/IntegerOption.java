/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import com.dsi.parallax.ml.util.bounds.*;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * IntegerOption is Integer Option
 * 
 * @author Josh Attenberg
 */
public class IntegerOption extends Option {

	private static final long serialVersionUID = -4385706421580269515L;
	// min, max, default, current values for integer options
	private final LowerBound lowerBound;
	private final UpperBound upperBound;
	private final int DEFAULT;

	/**
	 * Class constructor specifying short name, long name, description, default
	 * value, minimum,maximum and optimize flag to create
	 * 
	 * @param shortName
	 *            short name
	 * @param longName
	 *            long name
	 * @param desc
	 *            description
	 * @param min
	 *            minimum
	 * @param max
	 *            maximum
	 * @param defaultv
	 *            default value
	 * @param optimizable
	 *            optimize flag
	 */
	public IntegerOption(String shortName, String longName, String desc,
			int defaultv, boolean optimizable, LowerBound lowerBound,
			UpperBound upperBound) {
		super(shortName, longName, optimizable, desc);
		checkArgument(
				lowerBound.getBoundValue() < upperBound.getBoundValue(),
				"lower bound must be above upper bound, given lower: %s, and upper: %s",
				lowerBound.getBoundValue(), upperBound.getBoundValue());
		this.upperBound = upperBound;
		this.lowerBound = lowerBound;
		DEFAULT = checkCurrent(defaultv);
	}

	public IntegerOption(String shortName, String longName, String desc,
			int defaultv, boolean optimizable, LowerBound lowerBound) {
		this(shortName, longName, desc, defaultv, optimizable, lowerBound,
				new LessThanValueBound(Double.POSITIVE_INFINITY));
	}

	public IntegerOption(String shortName, String longName, String desc,
			int defaultv, boolean optimizable, UpperBound upperBound) {
		this(shortName, longName, desc, defaultv, optimizable,
				new GreaterThanValueBound(Double.NEGATIVE_INFINITY), upperBound);
	}

	public IntegerOption(String shortName, String longName, String desc,
			int defaultv, boolean optimizable) {
		this(shortName, longName, desc, defaultv, optimizable,
				new GreaterThanValueBound(Double.NEGATIVE_INFINITY),
				new LessThanValueBound(Double.POSITIVE_INFINITY));
	}

	/**
	 * The method check if the integer value between maximum and minimum
	 * 
	 * @param current
	 *            integer value
	 * @return integer value
	 */
	public int checkCurrent(int current) {
		checkArgument(
				lowerBound.satisfiesBound(current)
						&& upperBound.satisfiesBound(current),
				"input value (%s) outside options bounds, "
						+ lowerBound.representBound() + " and "
						+ upperBound.representBound(), current);
		return current;
	}

	public LowerBound getLowerBound() {
		return lowerBound;
	}

	public UpperBound getUpperBound() {
		return upperBound;
	}

	/**
	 * The method gets default integer value
	 * 
	 * @return integer value
	 */
	public int getDEFAULT() {
		return DEFAULT;
	}

	/**
	 * The method returns the class's Type "INTEGER"
	 * 
	 * @return INTEGER
	 */
	@Override
	public OptionType getType() {
		return OptionType.INTEGER;
	}
	
	public Bounds getBounds() {
		return new Bounds(lowerBound, upperBound);
	}

	/**
	 * The method copy all the values of IntegerOption to a new one
	 * 
	 * @return Option IntegerOption
	 */
	@Override
	public Option copyOption() {
		return new IntegerOption(shortName, longName, description, DEFAULT,
				isOptimizable, lowerBound, upperBound);
	}

	@Override
	public int hashCode() {
		if (hashcode == -1) {
			hashcode = super.hashCode();
			hashcode = hashcode * 17 + lowerBound.representBound().hashCode();
			hashcode = hashcode * 17 + upperBound.representBound().hashCode();
			hashcode = hashcode * 17 + DEFAULT;
		}
		return hashcode;
	}
}
