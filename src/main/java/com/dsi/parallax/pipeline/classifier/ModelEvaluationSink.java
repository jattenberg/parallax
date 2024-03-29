/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.classifier;

import com.dsi.parallax.ml.evaluation.OnlineEvaluation;
import com.dsi.parallax.pipeline.AbstractSink;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Iterators;

import java.util.Iterator;

import static com.google.common.base.Preconditions.checkArgument;

// TODO: Auto-generated Javadoc
/**
 * As input, gets a sequence of OnlineEvalutation objects,
 * returns the final example, getting the estimate of performance
 * over all the data. 
 * @author jattenberg
 *
 */
public class ModelEvaluationSink extends AbstractSink<OnlineEvaluation, OnlineEvaluation>{

	/** The source. */
	private Iterator<OnlineEvaluation> source;
	
	/**
	 * Instantiates a new model evaluation sink.
	 */
	public ModelEvaluationSink() {
		super();
	}
	
	/**
	 * sets the input pipeline, an interator over Context<OnlineEvaluation>.
	 *
	 * @param source the new source
	 */
	@Override
	public void setSource(Iterator<Context<OnlineEvaluation>> source) {
		checkArgument(null != source);
		this.source = Iterators.transform(source, uncontextifier);
	}

	/**
	 * returns the final OnlineEvaluation in the input stream.
	 *
	 * @return the online evaluation
	 */
	@Override
	public OnlineEvaluation next() {
		OnlineEvaluation out = null;
		while(source.hasNext())
			out = source.next();
		return out;
	}

	/**
	 * does the sink have another OnlineEvaluation?.
	 *
	 * @return true, if successful
	 */
	@Override
	public boolean hasNext() {
		return source == null ? false : source.hasNext();
	}

}
