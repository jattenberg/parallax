/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.instance;

import java.lang.reflect.Type;
import java.util.Iterator;

import com.dsi.parallax.ml.instance.Instance;
import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.pipeline.AbstractExpandingPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import com.google.gson.reflect.TypeToken;

// TODO: Auto-generated Javadoc
/**
 * The Class InstancesToInstanzePipe.
 */
public class InstancesToInstanzePipe extends AbstractExpandingPipe<Instances<?>, Instance<?>> {

    /** The Constant serialVersionUID. */
    private static final long serialVersionUID = -5878522922955659492L;
	
	/** The contextifier. */
	private ContextifyInstanzeFunction contextifier;

    /**
     * Instantiates a new instances to instanze pipe.
     */
    public InstancesToInstanzePipe() {
    	super();
    	this.contextifier = new ContextifyInstanzeFunction();
    }

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.Pipe#getType()
	 */
	@Override
	public Type getType() {
		return new TypeToken<InstancesToInstanzePipe>(){}.getType();
	}
	
	/**
	 * The Class ContextifyInstanzeFunction.
	 */
	private class ContextifyInstanzeFunction implements Function<Instance<?>, Context<Instance<?>>> {

		/* (non-Javadoc)
		 * @see com.google.common.base.Function#apply(java.lang.Object)
		 */
		@SuppressWarnings("unchecked")
		@Override
		public Context<Instance<?>> apply(Instance<?> inst) {
			Context<?> context = Context.createContext(inst.getID(), inst.getLabel()+"", inst);
			return (Context<Instance<?>>)context;
		}
	}

	/* (non-Javadoc)
	 * @see com.parallax.pipeline.AbstractExpandingPipe#operate(com.parallax.pipeline.Context)
	 */
	@Override
	protected Iterator<Context<Instance<?>>> operate(Context<Instances<?>> context) {
		Instances<?> insts = context.getData();
		return Iterators.transform(insts.iterator(), contextifier);
	}
}
