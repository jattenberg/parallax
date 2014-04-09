/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.target;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * BinaryClassificationTarget class extends the RealValuedTarget class 
 * to provide functionality for target for binary classification
 *
 * @author Josh Attenberg
 */
public class BinaryClassificationTarget extends RealValuedTarget {

    /**
     * Class constructor specifying real value to create
     * @param value real value
     */
	public BinaryClassificationTarget(double value) {
		super(value);
	}

    /**
     * The method set real value
     * @param value
     */
	@Override
	public void setValue(double value) {
		checkArgument(value >= 0 && value <= 1, "classifier value should be between 0 and 1, given: %s", value);
		super.setValue(value);
	}
}
