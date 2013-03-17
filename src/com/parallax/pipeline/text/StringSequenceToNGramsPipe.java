/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.List;

import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * StringSequenceToNGramsPipe is primitive class for getting List contents of String object into context
 *
 * @author Josh Attenberg
 */
public class StringSequenceToNGramsPipe extends AbstractPipe<List<String>,List<String>> {

    private static final long serialVersionUID = -4370637810974890776L;
    private int[] gramSizes = null;
    private final static String SEP = "_";
    private StringBuilder buff;

    /**
     * Class constructor specifying gram to create
     * @param gram gram
     */
    public StringSequenceToNGramsPipe(int gram) {
       	this(new int[]{gram});    
    }

    /**
     * Class constructor specifying multiple grams to create
     * @param grams multiple grams
     */
    public StringSequenceToNGramsPipe(int[] grams) {
    	super();
    	checkValidSizes(grams);
        gramSizes = grams;
        buff = new StringBuilder();
    }

    /**
     * The method returns the class's Type "StringSequenceToNGramsPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceToNGramsPipe>(){}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<List<String>> context) {
        List<String> payload = context.getData();
        List<String> out = Lists.newArrayList();
        buff.setLength(0);
        
        for(int i = 0; i < payload.size(); i++) {
            String token = payload.get(i);
            for(int len : gramSizes) {
                if(len > i+1) continue;
                if(len == 1){ out.add(token); continue; }
                buff.setLength(0);
                
                for(int k = len-1; k > 0; k--)
                    buff.append( payload.get(i-k) + SEP);
                buff.append(token);
                out.add(buff.toString());
            }
        }  
        return Context.createContext(context, out);
	}
}
