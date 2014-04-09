/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.classifier;

import java.lang.reflect.Type;

import com.dsi.parallax.ml.classifier.Classifier;
import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

// TODO: Auto-generated Javadoc
/**
 * The Class PredictionPipe.
 *
 * @param <C> the generic type
 */
public class PredictionPipe<C extends Classifier<C>> extends AbstractPipe<BinaryClassificationInstance, BinaryClassificationTarget> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 1772129622702182015L;
	
	/** The model. */
	private final Classifier<C> model;
	
	/**
	 * Instantiates a new prediction pipe.
	 *
	 * @param model the model
	 */
	public PredictionPipe(Classifier<C> model) {
		super();
		this.model = model;
	}
	
	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<PredictionPipe<C>>(){}.getType();
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Context<BinaryClassificationTarget> operate(
			Context<BinaryClassificationInstance> context) {
		BinaryClassificationInstance inst = context.getData();
		return Context.createContext(context, model.predict(inst));
	}

}
