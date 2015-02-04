/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.instance;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.pipeline.AbstractSink;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Iterators;

import java.util.Iterator;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Pipe used for accumulating BinaryClassificationInstance and condensing them
 * into a single BinaryClassificationInstances
 */
public class BinaryClassificationInstanceSink
		extends
		AbstractSink<BinaryClassificationInstance, BinaryClassificationInstance> {

	/** iterator providing BinaryClassificationInstance */
	private Iterator<BinaryClassificationInstance> source;

	/**
	 * Instantiates a new binary classification instance sink.
	 */
	public BinaryClassificationInstanceSink() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Sink#next()
	 */
	@Override
	public BinaryClassificationInstance next() {
		return source.next();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Sink#hasNext()
	 */
	@Override
	public boolean hasNext() {
		return source == null ? false : source.hasNext();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Sink#setSource(java.util.Iterator)
	 */
	@Override
	public void setSource(Iterator<Context<BinaryClassificationInstance>> source) {
		checkArgument(null != source);
		this.source = Iterators.transform(source, uncontextifier);
	}

}
