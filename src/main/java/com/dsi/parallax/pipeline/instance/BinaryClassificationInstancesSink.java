/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.instance;

import java.util.Iterator;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.AbstractSink;
import com.dsi.parallax.pipeline.Context;

/**
 * A sink that reads from a pipeline emitting Contexts containing
 * {@link BinaryClassificationInstance}'s, collects them and emits a iterator
 * over a single collection, a {@link BinaryClassificationInstances}
 * 
 * @author jattenberg
 */
public class BinaryClassificationInstancesSink
		extends
		AbstractSink<BinaryClassificationInstance, BinaryClassificationInstances> {

	/** The instances to be iterated over and collected */
	private BinaryClassificationInstanceSink instanzes;

	/**
	 * Instantiates a new binary classification instances sink.
	 */
	public BinaryClassificationInstancesSink() {
		super();
		instanzes = new BinaryClassificationInstanceSink();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Sink#next()
	 */
	@Override
	public BinaryClassificationInstances next() {

		BinaryClassificationInstances out = null;
		while (instanzes.hasNext()) {
			BinaryClassificationInstance inst = instanzes.next();
			if (null == out)
				out = new BinaryClassificationInstances(inst.size());
			out.addInstance(inst);
		}
		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Sink#setSource(java.util.Iterator)
	 */
	@Override
	public void setSource(Iterator<Context<BinaryClassificationInstance>> source) {
		instanzes.setSource(source);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Sink#hasNext()
	 */
	@Override
	public boolean hasNext() {
		return instanzes.hasNext();
	}

}
