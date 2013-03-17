/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import java.lang.reflect.Type;
import java.util.Collection;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

/**
 * StringSequenceConcatingCombiner combines multiple list context of String to one list context of String
 *
 * @author Josh Attenberg
 */
public class StringSequenceConcatingCombiner extends AbstractCombiner<List<String>, List<String>>{

    private static final long serialVersionUID = -4534498494140271842L;

    /**
     * The method iterates multiple context and combines them to one context
     * @param branchOutput multiple context
     * @return context
     */
    @Override
    public Context<List<String>> combineExamples(
            Collection<Context<List<String>>> branchOutput) {
        List<String> combined = Lists.newLinkedList();
        String id = null;
        String label = null;
        
        for(Context<List<String>> context : branchOutput) {
            id = context.id==null?id:context.id;
            label = context.label==null?label:context.label;
            combined.addAll(context.getData());
        }
        
        return new Context<List<String>>(id, label, combined);
    }

    /**
     * The method returns the class's Type "StringSequenceConcatingCombiner"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceConcatingCombiner>(){}.getType();
	} 
}
