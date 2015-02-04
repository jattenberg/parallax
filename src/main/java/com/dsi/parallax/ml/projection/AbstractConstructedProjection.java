/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.projection;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.ml.vector.LinearVector;
import com.google.common.collect.Lists;

import java.util.Collection;

/**
 * Base class for Abstract constructed projections, projections that consider
 * example data and require some kind of training (construction) process. For
 * instance, Singular Value decomposition
 * 
 * @author jattenberg
 */
public abstract class AbstractConstructedProjection extends AbstractProjection
		implements ConstructedProjection {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 649765443146273563L;

	/**
	 * Instantiates a new abstract constructed projection.
	 * 
	 * @param inDim
	 *            - the number of dimensions for inbound vectors
	 * @param outDim
	 *            - the number of dimensions for output vectors; the size of the
	 *            output space.
	 */
	protected AbstractConstructedProjection(int inDim, int outDim) {
		super(inDim, outDim);
	}

	/**
	 * Builds the internal data structures required for projection; the
	 * "training" process.
	 * 
	 * @param X
	 *            instances constituting the training data used for projection
	 *            construction
	 */
	public void build(Instances<?> X) {
		Collection<LinearVector> vecs = Lists.newLinkedList();
		for (Instance<?> inst : X)
			vecs.add(inst.getFeatureValues());
		build(vecs);
	}

}
