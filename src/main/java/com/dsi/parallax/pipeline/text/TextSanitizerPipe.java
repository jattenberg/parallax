/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

/**
 * TextSanitizerPipe is primitive class for getting contents of String object into context
 *
 * @author Josh Attenberg
 */
public class TextSanitizerPipe extends AbstractPipe<String,String> {

    private static final long serialVersionUID = -8813779890998243436L;

    /**
     * Class constructor
     */
    public TextSanitizerPipe() {
    	super();
    }

    /**
     * The method returns the class's Type "TextSanitizerPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<TextSanitizerPipe>(){}.getType();
	}

	@Override
	protected Context<String> operate(Context<String> context) {
        String payload = context.getData();
        String out = MLUtils.cleanText(payload, false);
        return Context.createContext(context, out);
	}

}
