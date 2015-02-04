/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.instance;

import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.Source;
import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Iterator;

/**
 * A pipeline source that provides BinaryClassificationInstances; can be used to
 * take BinaryClassificationInstance generated in a variety of manners and use
 * them in the context of pipeline systems.
 */
public class BinaryClassificationInstanceSource implements
		Source<BinaryClassificationInstances> {

	/** The Constant serialVersionUID. */
	private static final long serialVersionUID = 5705248251822202525L;

	/** The instances to be iterated over. */
	private BinaryClassificationInstances insts;

	/**
	 * Instantiates a new binary classification instance source.
	 * 
	 * @param insts
	 *            the instances to be iterated over.
	 */
	public BinaryClassificationInstanceSource(
			BinaryClassificationInstances insts) {
		this.insts = insts;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Source#provideData()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Iterator<Context<BinaryClassificationInstances>> provideData() {
		return Iterators.forArray(Context.createContext(insts));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see com.parallax.pipeline.Source#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<BinaryClassificationInstanceSource>() {
		}.getType();
	}

}
