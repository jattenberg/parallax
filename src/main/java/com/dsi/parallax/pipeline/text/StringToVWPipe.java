/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.ml.util.VW;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

/**
 * StringToVWPipe is primitive class for getting contents of String object into context
 *
 * @author Josh Attenberg
 */
public class StringToVWPipe extends AbstractPipe<String, VW> {

    private static final long serialVersionUID = 1L;

    /**
     * Class constructor
     */
    public StringToVWPipe() {
        super();
    }

    /**
     * The method returns the class's Type "StringToVWPipe"
     * @return Type
     */
    @Override
    public Type getType() {
        return new TypeToken<StringToVWPipe>(){}.getType();
    }

	@Override
	protected Context<VW> operate(Context<String> context) {
        String line = context.getData();
        VW vw = VW.fromVWLine(line);
        return Context.createContext(context, vw);
	}
}
