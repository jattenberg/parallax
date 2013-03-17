/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.classifier;

import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.target.BinaryClassificationTarget;
import com.parallax.ml.util.MLUtils;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * A simple pipe that adds made up (random) binary labels to instances.
 * 
 * @author jattenberg
 */
public class BinaryLabelFakerPipe
		extends
		AbstractPipe<BinaryClassificationInstance, BinaryClassificationInstance> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1L;

	/**
	 * Instantiates a new binary label faker pipe.
	 */
	public BinaryLabelFakerPipe() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<BinaryLabelFakerPipe>() {
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
			Context<BinaryClassificationInstance> context) {
		BinaryClassificationInstance inst = context.getData();
		double label = MLUtils.GENERATOR.nextDouble() < .5 ? 0 : 1;
		inst.setLabel(new BinaryClassificationTarget(label));
		return Context.createContext(context, inst, inst.getID(), "" + label);
	}

}
