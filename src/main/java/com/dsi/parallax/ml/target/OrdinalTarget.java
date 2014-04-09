/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import java.util.List;

/**
 * OrdinalTarget defines ordinal target
 *
 * @author Josh Attenberg
 */
public class OrdinalTarget extends MultiClassTarget {

    /**
     * Class constructor specifying number of classes and multiple values to create
     * @param numClasses number of classes
     * @param values multiple value
     */
	public OrdinalTarget(int numClasses, List<Double> values) {
		super(numClasses, values);
	}

    /**
     * Class constructor specifying number of classes and multiple values to create
     * @param numClasses number of classes
     * @param values  multiple value
     */
	public OrdinalTarget(int numClasses, double[] values) {
		super(numClasses, values);
	}
}
