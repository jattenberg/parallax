/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.distributions;


/**
 * An enumeration over the types of multinomial distributions
 */
public enum DistributionType {

	/** The multinomial. */
	MULTINOMIAL, 
	
	/** The bernoulli. */
	BERNOULLI, 
	
	/** The gaussian. */
	GAUSSIAN, 
	
	/** The kde. */
	KDE,
	
	/** The histogram. */
	HISTOGRAM

}
