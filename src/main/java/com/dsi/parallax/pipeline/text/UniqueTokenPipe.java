/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.List;
import java.util.Set;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.gson.reflect.TypeToken;

/**
 * pipe extracting only the unique tokens.

 * @author jattenberg
 *
 */
public class UniqueTokenPipe extends AbstractPipe<List<String>,List<String>> {

	private static final long serialVersionUID = -1503672116466929071L;

    /**
     * Class constructor
     */
	public UniqueTokenPipe() {
		super();
	}

    /**
     * The method returns the class's Type "UniqueTokenPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<UniqueTokenPipe>(){}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<List<String>> context) {
		Set<String> uniqueTokens = Sets.newLinkedHashSet(context.getData());
		List<String> outData = Lists.newLinkedList(uniqueTokens);
		return Context.createContext(context, outData);	
	}

}
