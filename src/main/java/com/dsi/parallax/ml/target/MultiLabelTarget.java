/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import org.apache.commons.lang.builder.EqualsBuilder;

import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * MultiLabelTarget defines multiple label target
 * 
 * @author Josh Attenberg
 */
public class MultiLabelTarget implements Target, Comparable<Target> {
	protected final int numClasses;
	protected List<BinaryClassificationTarget> values;

	/**
	 * Class constructor specifying number of classes and
	 * BinaryClassificationTarget value to create
	 * 
	 * @param numClasses
	 *            number of classes
	 * @param values
	 *            BinaryClassificationTarget values
	 */
	public MultiLabelTarget(int numClasses, List<Double> values) {
		this.numClasses = numClasses;
		copyTargetValues(values);
	}

	/**
	 * Class constructor specifying number of classes and
	 * BinaryClassificationTarget value to create
	 * 
	 * @param numClasses
	 *            number of classes
	 * @param values
	 *            array multiple BinaryClassificationTarget values
	 */
	public MultiLabelTarget(int numClasses, double[] values) {
		this.numClasses = numClasses;
		copyTargetValues(values);
	}

	protected void copyTargetValues(List<Double> candidateValues) {
		checkArgument(candidateValues.size() == numClasses);
		this.values = Lists.newArrayList();
		for (double val : candidateValues)
			this.values.add(new BinaryClassificationTarget(val));
	}

	protected void copyTargetValues(double[] candidateValues) {
		checkArgument(candidateValues.length == numClasses);
		this.values = Lists.newArrayList();
		for (double val : candidateValues)
			this.values.add(new BinaryClassificationTarget(val));
	}

	/**
	 * The method gets list of BinaryClassificationTarget values
	 * 
	 * @return list of BinaryClassificationTarget value
	 */
	public List<BinaryClassificationTarget> getValues() {
		return values;
	}

	/**
	 * The method sets list of BinaryClassificationTarget values
	 * 
	 * @param values
	 *            list of BinaryClassificationTarget values
	 */
	public void setValues(List<BinaryClassificationTarget> values) {
		this.values = values;
	}

	@Override
	public String toString() {
		return "[" + Joiner.on(", ").join(values) + "]";
	}

	@Override
	public int compareTo(Target t) {
		if (t instanceof MultiLabelTarget || t instanceof OrdinalTarget
				|| t instanceof MultiClassTarget) {
			MultiLabelTarget o = (MultiLabelTarget) t;
			if (numClasses < o.numClasses)
				return -1;
			else if (o.numClasses < numClasses)
				return 1;
			else {
				for (int i = 0; i < numClasses; i++) {
					int comp = values.get(i).compareTo(o.values.get(i));
					if (comp != 0)
						return comp;
				}
				return 0;
			}
		} else
			return 1;
	}

	@Override
	public boolean equals(Object o) {
		if (!o.getClass().isAssignableFrom(this.getClass())) {
			return false;
		} else {
			MultiLabelTarget t = (MultiLabelTarget) o;
			if (numClasses != t.numClasses)
				return false;
			EqualsBuilder builder = new EqualsBuilder();
			for (int i = 0; i < numClasses; i++)
				builder.append(values.get(i), t.values.get(i));
			return builder.isEquals();
		}
	}
}
