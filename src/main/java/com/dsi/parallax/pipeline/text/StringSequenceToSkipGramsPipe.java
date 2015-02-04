/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Iterator;
import java.util.List;

/**
 * StringSequenceToSkipGramsPipe is primitive class for getting List contents of String object into context
 *
 * @author Josh Attenberg
 */
public class StringSequenceToSkipGramsPipe extends AbstractPipe<List<String>,List<String>> {

    private static final long serialVersionUID = -4370637810974890776L;
    private int[] gramSizes = null;
    private final static String SEP = "_";
    StringBuilder buff;

    /**
     * Class constructor specifying gram to create
     * @param gram gram
     */
    public StringSequenceToSkipGramsPipe(int gram) {
        this(new int[]{gram});
    }

    /**
     * Class constructor specifying multiple grams to create
     * @param grams multiple grams
     */
    public StringSequenceToSkipGramsPipe(int[] grams) {
    	super();
    	checkValidSizes(grams);
        gramSizes = grams;
        buff = new StringBuilder();
    }

    /**
     * The method returns iterator object which includes contexts of string list
     * @param source iterator of contexts of string list
     * @return iterator of contexts of string list
     */
	@Override
	public Iterator<Context<List<String>>> processIterator(
			Iterator<Context<List<String>>> source) {
		return Iterators.transform(source, function);
	}

    /**
     * The method returns the class's Type "StringSequenceToSkipGramsPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceToSkipGramsPipe>(){}.getType();
	}


	@Override
	protected Context<List<String>> operate(Context<List<String>> context) {
        List<String> payload = context.getData();
        List<String> out = Lists.newArrayList();
        
        buff.setLength(0);
        
        for(int i = 0; i < payload.size(); i++) {
            String token = payload.get(i);
            for(int len : gramSizes) {
                if(len <= 0 || len > i+1) continue;
                if(len == 1){ out.add(token); continue; }
                buff.setLength(0);
                buff.append(payload.get(i-(len-1)));
                buff.append(SEP + token);
                out.add(buff.toString());
            }
        }  
        return Context.createContext(context, out);
	}


}
