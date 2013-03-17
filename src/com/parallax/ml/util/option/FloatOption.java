/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

import static com.google.common.base.Preconditions.checkArgument;

import com.parallax.ml.util.MLUtils;
import com.parallax.ml.util.bounds.Bounds;
import com.parallax.ml.util.bounds.GreaterThanValueBound;
import com.parallax.ml.util.bounds.LessThanValueBound;
import com.parallax.ml.util.bounds.LowerBound;
import com.parallax.ml.util.bounds.UpperBound;

/**
 * FloatOption is Float Option
 * 
 * @author Josh Attenberg
 */
public class FloatOption extends Option {

	private static final long serialVersionUID = -4385706421580269515L;
	// min, max, default, current values for float options
	private final LowerBound lowerBound;
	private final UpperBound upperBound;
	private final double DEFAULT;

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
	 * @param defaultv
	 *            default value
	 * @param optimizable
	 *            optimize flag
	 */
	public FloatOption(String shortName, String longName, String desc,
			double defaultv, boolean optimizable, LowerBound lowerBound,
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

	public FloatOption(String shortName, String longName, String desc,
			double defaultv, boolean optimizable, LowerBound lowerBound) {
		this(shortName, longName, desc, defaultv, optimizable, lowerBound,
				new LessThanValueBound(Double.POSITIVE_INFINITY));
	}

	public FloatOption(String shortName, String longName, String desc,
			double defaultv, boolean optimizable, UpperBound upperBound) {
		this(shortName, longName, desc, defaultv, optimizable,
				new GreaterThanValueBound(Double.NEGATIVE_INFINITY), upperBound);
	}

	public FloatOption(String shortName, String longName, String desc,
			double defaultv, boolean optimizable) {
		this(shortName, longName, desc, defaultv, optimizable,
				new GreaterThanValueBound(Double.NEGATIVE_INFINITY),
				new LessThanValueBound(Double.POSITIVE_INFINITY));
	}

	/**
	 * The method check if the float value between maximum and minimum
	 * 
	 * @param current
	 *            float value
	 * @return float value
	 */
	public double checkCurrent(double current) {
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
	 * The method gets default float value
	 * 
	 * @return default float value
	 */
	public double getDEFAULT() {
		return DEFAULT;
	}

	/**
	 * The method returns the class's Type "FLOAT"
	 * 
	 * @return FLOAT
	 */
	@Override
	public OptionType getType() {
		return OptionType.FLOAT;
	}
	
	public Bounds getBounds() {
		return new Bounds(lowerBound, upperBound);
	}

	/**
	 * The method copy all the values of FloatOption to a new one
	 * 
	 * @return Option FloatOption
	 */
	@Override
	public Option copyOption() {
		return new FloatOption(shortName, longName, description, DEFAULT,
				isOptimizable, lowerBound, upperBound);
	}
	
	@Override
	public int hashCode() {
		if (hashcode == -1) {
			hashcode = super.hashCode();
			hashcode = hashcode * 17 + lowerBound.representBound().hashCode();
			hashcode = hashcode * 17 + upperBound.representBound().hashCode();
			hashcode = hashcode * 17 + MLUtils.hashDouble(DEFAULT);
		}
		return hashcode;
	}

}
