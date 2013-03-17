/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;

/**
 * replaces every incoming example with the O defined on construction
 * @author jattenberg
 *
 * @param <O>
 * @author Josh Attenberg
 */
public class ReplacementPipe<I,O> extends AbstractPipe<I,O> {

    private static final long serialVersionUID = 8085330508962305354L;
    private final Context<O> dummy;

    /**
     * Class constructor specifying object to create
     * @param obj incoming example
     */
    public ReplacementPipe(O obj) {
    	super();
        dummy = Context.createContext(obj);
    }

    /**
     * The method returns the class's Type "ReplacementPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<ReplacementPipe<I,O>>(){}.getType();
	}

	@Override
	protected Context<O> operate(Context<I> in) {
		return dummy;
	}
}
