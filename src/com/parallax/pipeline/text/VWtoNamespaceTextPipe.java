/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import java.lang.reflect.Type;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.util.VW;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * VWtoNamespaceTextPipe is primitive class for getting contents of VW object into context
 *
 * @author Josh Attenberg
 */
public class VWtoNamespaceTextPipe extends AbstractPipe<VW, NamespaceText> {

	private static final long serialVersionUID = 3576890784097532193L;

    /**
     * Class constructor
     */
	public VWtoNamespaceTextPipe() {
		super();
	}

    /**
     * The method returns the class's Type "VWtoNamespaceTextPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<VWtoNamespaceTextPipe>() {
		}.getType();
	}

	@Override
	protected Context<NamespaceText> operate(Context<VW> context) {
		VW vw = context.getData();
		NamespaceText nst = new NamespaceText(vw);
		Context<NamespaceText> out = Context.createContext(context, nst);
		out.setId(vw.getID());
		out.setLabel(vw.getCategory());
		return out;
	}

}
