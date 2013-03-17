/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.trees;

import org.apache.commons.lang.builder.EqualsBuilder;

import com.google.common.collect.ComparisonChain;
import com.parallax.ml.target.Target;

/**
 * A simple container for associating an attribute value with a label.
 * used for decision tree construction.
 * 
 * @param <T> the type of label being considered
 * @author jattenberg
 */
public class AttributeValueLabel<T extends Target> implements
		Comparable<AttributeValueLabel<T>> {

	/** The value of the attribute being considered */
	private final double value;
	
	/** The label associated with the attribute value */
	private final T label;

	/**
	 * Instantiates a new attribute value label.
	 *
	 * @param value The value of the attribute being considered
	 * @param label The label associated with the attribute value
	 */
	public AttributeValueLabel(double value, T label) {
		super();
		this.value = value;
		this.label = label;
	}

	/**
	 * Gets The value of the attribute being considered
	 *
	 * @return the value
	 */
	public double getValue() {
		return value;
	}

	/**
	 * Gets The label associated with the attribute value
	 *
	 * @return the label
	 */
	public T getLabel() {
		return label;
	}

	/* (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	@Override
	public int compareTo(AttributeValueLabel<T> o) {
		return ComparisonChain.start().compare(value, o.value)
				.compare(label, o.label).result();
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#equals(java.lang.Object)
	 */
	@Override
	public boolean equals(Object o) {
		if (o instanceof AttributeValueLabel) {
			@SuppressWarnings("rawtypes")
			AttributeValueLabel a = (AttributeValueLabel) o;
			if (label.getClass().isInstance(a.label)) {
				@SuppressWarnings("unchecked")
				AttributeValueLabel<T> b = (AttributeValueLabel<T>) a;
				return new EqualsBuilder().append(value, b.value)
						.append(label, b.label).isEquals();
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

}
