/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

/**
 * BooleanOption is Boolean Option
 *
 * @author Josh Attenberg
 */
public class BooleanOption extends Option {

	private static final long serialVersionUID = 9054810884610559910L;
	// default and current values for boolean options
	private final boolean DEFAULTB;

    /**
     * Class constructor specifying short name, long name, description, default value
     * and optimize flag to create
     * @param shortName short name
     * @param longName long name
     * @param desc description
     * @param defaultv default value
     * @param optimizable optimize flag
     */
	public BooleanOption(String shortName, String longName, String desc,
			boolean defaultv, boolean optimizable) {
		super(shortName, longName, optimizable, desc);
		DEFAULTB = defaultv;
	}

    /**
     * The method returns the class's Type "BOOLEAN"
     * @return OptionType
     */
	@Override
	public OptionType getType() {
		return OptionType.BOOLEAN;
	}

    /**
     * The method checks if the default value is true or false
     * @return
     */
	public boolean isDEFAULTB() {
		return DEFAULTB;
	}

    /**
     * The method gets default value
     * @return default value
     */
	public boolean getDEFAULTB() {
		return DEFAULTB;
	}

    /**
     * The method copy all the values of BooleanOption to a new one
     * @return Option BooleanOption
     */
	@Override
	public Option copyOption() {
		return new BooleanOption(shortName, longName, description, DEFAULTB, isOptimizable);
	}
	
	@Override
	public int hashCode() {
		if(hashcode == -1) {
			hashcode = super.hashCode();
			hashcode = hashcode * 17*(DEFAULTB ? 1 : 2);
		}		
		return hashcode;
	}
}
