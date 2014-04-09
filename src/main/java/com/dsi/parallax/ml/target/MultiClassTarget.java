/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.List;

import com.dsi.parallax.ml.util.MLUtils;

/**
 * MultiClassTarget defines multiple class target
 *
 * @author Josh Attenberg
 */
public class MultiClassTarget extends MultiLabelTarget {

    /**
     * Class constructor specifying number of classes and  multiple target value to create
     * @param numClasses number of classes
     * @param values multiple target value
     */
	public MultiClassTarget(int numClasses, double[] values) {
		super(numClasses, values);
	}

    /**
     * Class constructor specifying number of classes and  multiple value to create
     * @param numClasses number of classes
     * @param values multiple value
     */
	public MultiClassTarget(int numClasses, List<Double> values) {
		super(numClasses, values);
	}
	
	protected void copyTargetValues(List<Double> candidateValues) {
		checkArgument(MLUtils.floatingPointEquals(1.0,MLUtils.sum(candidateValues)));
		super.copyTargetValues(candidateValues);
	}

	protected void copyTargetValues(double[] candidateValues) {
		checkArgument(MLUtils.floatingPointEquals(1.0,MLUtils.sum(candidateValues)));
		super.copyTargetValues(candidateValues);
	}
}
