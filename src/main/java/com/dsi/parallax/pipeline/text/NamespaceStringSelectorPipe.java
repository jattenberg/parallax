/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

/**
 * NamespaceStringSelectorPipe is primitive class for getting contents of NamespaceText object into context
 *
 * @author Josh Attenberg
 */
public class NamespaceStringSelectorPipe extends AbstractPipe<NamespaceText, String> {

    private static final long serialVersionUID = -3321275713779195819L;
    private final String namespace;

    /**
     * Class constructor specifying namespace to create
     * @param namespace namespace
     */
    public NamespaceStringSelectorPipe(String namespace) {
        super();
    	this.namespace = namespace;
    }

    /**
     * The method returns the class's Type "NamespaceStringSelectorPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<NamespaceStringSelectorPipe>(){}.getType();
	}

	@Override
	protected Context<String> operate(Context<NamespaceText> context) {
        NamespaceText nst = context.getData();
        String out = "";
        if(nst.containsNamespace(namespace))
            out=nst.getTextForNamespace(namespace);
        
        return Context.createContext(context, out);

	}
}
