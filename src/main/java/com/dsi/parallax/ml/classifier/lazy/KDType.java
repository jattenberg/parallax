/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.classifier.lazy;

/**
 * The distance function used for kd trees
 */
public enum KDType {

	/** The manhattan. */
	MANHATTAN,
	/** The weightedmanhattan. */
	WEIGHTEDMANHATTAN,
	/** The euclidian. */
	EUCLIDIAN,
	/** The weightedeuclidian. */
	WEIGHTEDEUCLIDIAN
}
