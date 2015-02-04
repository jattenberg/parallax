/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.vector.LinearVector;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Base class for Abstract projections, methods that transform data from one
 * space to another.
 * 
 * @author jattenberg
 */
public abstract class AbstractProjection implements Projection {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 8591964023986636692L;

	/** the number of dimensions for inbound vectors. */
	protected final int inDim;
	/**
	 * the number of dimensions for output vectors; the size of the output
	 * space.
	 */
	protected final int outDim;

	/**
	 * Instantiates a new abstract projection.
	 * 
	 * @param inDim
	 *            - the number of dimensions for inbound vectors
	 * @param outDim
	 *            - the number of dimensions for output vectors; the size of the
	 *            output space.
	 */
	protected AbstractProjection(int inDim, int outDim) {
		checkArgument(inDim > 0, "input dimensino must be positive, given %s",
				inDim);
		checkArgument(outDim > 0,
				"output dimensino must be positive, given %s", outDim);
		this.inDim = inDim;
		this.outDim = outDim;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.ml.projection.Projection#project(com.parallax.ml.instance
	 * .Instanze)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public <I extends Instance<?>> I project(I x) {
		LinearVector lv = project(x.getFeatureValues());
		return (I) x.cloneNewVector(lv);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.projection.Projection#getInputDimension()
	 */
	@Override
	public int getInputDimension() {
		return inDim;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.ml.projection.Projection#getOutputDimension()
	 */
	@Override
	public int getOutputDimension() {
		return outDim;
	}

}
