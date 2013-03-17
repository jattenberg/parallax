/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.instance;

import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.target.DummyTargetParser;
import com.parallax.ml.target.TargetParser;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * a pipe that makes binary classification instances from linear vectors and
 * string labels in the context object.
 * 
 * The linear vector undergoes no manipulation; it's feature values are taken
 * literally as the output instances feature values.
 * 
 * The input context's id (if any!) is taken literally to be output instance's
 * id
 * 
 * The String label in the input context is transformed to the output instances
 * binary label through the use of a TargetParser {@link TargetParser}
 * 
 * @author jattenberg
 */
public class BinaryInstancesFromVectorPipe extends
		AbstractPipe<LinearVector, BinaryClassificationInstance> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 4592209650013108791L;

	/**
	 * The parser used to transform string labels into
	 * BinaryClassificationTargets
	 */
	private final TargetParser<? extends BinaryClassificationTarget> parser;

	/**
	 * Instantiates a new binary instances from vector pipe.
	 * 
	 * this default constructor uses a DummyTargetParser
	 * {@link DummyTargetParser} to transform string labels into
	 * {@link BinaryClassificationInstance}s
	 * 
	 */
	public BinaryInstancesFromVectorPipe() {
		this(new DummyTargetParser<BinaryClassificationTarget>());
	}

	/**
	 * Instantiates a new binary instances from vector pipe.
	 * 
	 * @param parser
	 *            The parser used to transform string labels into
	 *            BinaryClassificationTargets
	 */
	public BinaryInstancesFromVectorPipe(
			TargetParser<? extends BinaryClassificationTarget> parser) {
		super();
		this.parser = parser;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<BinaryInstancesFromVectorPipe>() {
		}.getType();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<BinaryClassificationInstance> operate(
			Context<LinearVector> context) {
		LinearVector payload = context.getData();
		BinaryClassificationInstance inst = new BinaryClassificationInstance(
				payload);
		if (null != context.getId())
			inst.setID(context.getId());
		if (null != context.getLabel()) {
			String stringLabel = context.getLabel();
			BinaryClassificationTarget target = parser.parseTarget(stringLabel);
			if (null != target)
				inst.setLabel(target);
		}
		return Context.createContext(context, inst);

	}
}
