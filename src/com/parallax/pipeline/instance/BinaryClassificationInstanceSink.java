/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.instance;

import static com.google.common.base.Preconditions.checkArgument;

import java.util.Iterator;

import com.google.common.collect.Iterators;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.pipeline.AbstractSink;
import com.parallax.pipeline.Context;

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
