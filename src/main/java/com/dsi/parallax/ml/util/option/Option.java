/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util.option;

import java.io.Serializable;

/**
 * a pretty weak class for handling options in ML code TODO: use generics, etc
 * 
 * @author jattenberg
 * 
 */
public abstract class Option implements Serializable {

	private static final long serialVersionUID = -5069442196993728148L;
	protected final String shortName;
	protected final String longName;
	protected final String description;
	protected final boolean isOptimizable;
	protected int hashcode = -1;

	protected Option(String shortName, String longName, boolean optimizable,
			String desc) {
		this.shortName = shortName;
		this.longName = longName;
		this.isOptimizable = optimizable;
		this.description = desc != null ? desc : "";
	}

    /**
     * The method gets short name
     * @return short name
     */
	public String getShortName() {
		return shortName;
	}

    /**
     * The method gets long name
     * @return
     */
	public String getLongName() {
		return longName;
	}

    /**
     * The method is abstract to get OptionType
     * @return OptionType
     */
	public abstract OptionType getType();

    /**
     * The method gets description
     * @return description
     */
    public String getDescription() {
		return description;
	}

    /**
     * The method is abstract to copy all of values to new Option object or instance
     * @return Option
     */
	public abstract Option copyOption();
	
	@Override
	public int hashCode() {
		if(hashcode == -1) {
			hashcode = 1;
			hashcode = hashcode * 17 + shortName.hashCode();
			hashcode = hashcode * 17 + longName.hashCode();
			hashcode = hashcode * 17 + description.hashCode();
			hashcode = hashcode * 17*(isOptimizable ? 1 : 2);
		}		
		return hashcode;
	}
}
