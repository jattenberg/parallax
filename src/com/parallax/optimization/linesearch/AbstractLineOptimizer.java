/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.optimization.linesearch;

import org.apache.log4j.Logger;

public abstract class AbstractLineOptimizer implements LineOptimizer {
	protected final Logger logger = Logger.getLogger(this.getClass());
}
