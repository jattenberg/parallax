/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util.option;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.EnumSet;
import java.util.List;

import com.google.common.collect.Lists;

/**
 * EnumOption is enum Option
 *
 * @param <T> generic type
 * @author Josh Attenberg
 */
public class EnumOption<T extends Enum<T>> extends Option {

	private static final long serialVersionUID = 8428875536430386528L;
	protected final EnumSet<T> valueSet;
	protected final T DEFAULTE;
	protected final Class<T> enumclass;

    /**
     * Class constructor specifying short name, long name, description, default value,
     * optimize flag and enum class to create
     * @param shortName short name
     * @param longName long name
     * @param optimizable optimize flag
     * @param desc description
     * @param enumclass enum class
     * @param defaulte default value
     */
	public EnumOption(String shortName, String longName, boolean optimizable,
			String desc, Class<T> enumclass, T defaulte) {
		super(shortName, longName, optimizable, desc);
		valueSet = EnumSet.allOf(enumclass);
		DEFAULTE = checkCurrent(defaulte);
		this.enumclass = enumclass;
	}

    /**
     * The method check if the value set contains enum class or not
     * @param current enum class
     * @return Enum enum class
     */
	@SuppressWarnings("unchecked")
	public T checkCurrent(Enum<T> current) {
		checkArgument(valueSet.contains(current), "%s isnt in enum set: %s", current, valueSet.toString());
		return (T)current;
	}

    /**
     * The method gets enum class set
     * @return enum class collection
     */
	public EnumSet<T> getValueSet() {
		return valueSet;
	}

    /**
     * The method get default value
     * @return enum class
     */
	public T getDEFAULTE() {
		return DEFAULTE;
	}

    /**
     * The method returns the class's Type "ENUM"
     * @return OptionType
     */
	@Override
	public OptionType getType() {
		return OptionType.ENUM;
	}

	/**
	 * get t
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public T[] getValues() {
		
		return (T[])valueSet.toArray();
	}
	
	public List<T> getValueList() {
		List<T> out = Lists.newArrayList(valueSet);
		return out;
	}
	
    /**
     * The method copy all the values of EnumOption to a new one
     * @return Option EnumOption
     */
	@Override
	public Option copyOption() {
		return new EnumOption<T>(shortName, longName, isOptimizable, description, enumclass, DEFAULTE);
	}
}
