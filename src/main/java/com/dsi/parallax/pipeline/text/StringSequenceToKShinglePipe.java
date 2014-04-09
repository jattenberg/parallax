/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.List;

import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Lists;
import com.google.gson.reflect.TypeToken;

/**
 * StringSequenceToKShinglePipe is primitive class for getting List contents of String object into context
 *
 * @author Josh Attenberg
 */
public class StringSequenceToKShinglePipe extends AbstractPipe<List<String>,List<String>> {

    private static final long serialVersionUID = -4370637810974890776L;
    private int[] shingleSizes = null;
    private final static String SEP = "_";
	StringBuilder buff;

    /**
     * Class constructor specifying shingle size to create
     * @param shingleSize shingle size
     */
    public StringSequenceToKShinglePipe(int shingleSize) {
         this(new int[]{shingleSize});
    }

    /**
     * Class constructor specifying multiple shingle sizes to create
     * @param sizes multiple shingle sizes
     */
    public StringSequenceToKShinglePipe(int[] sizes) {
    	super();
    	checkValidSizes(sizes);
        shingleSizes = sizes;
        buff = new StringBuilder();
    }
    
    private static List<String> buildCharList(List<String> input) {
        List<String> chars = Lists.newLinkedList();
        for(String term : input) {
            for(char token : term.toCharArray())
                chars.add(String.valueOf(token));
            chars.add(SEP);
        }
        return chars;
    }

    /**
     * The method returns the class's Type "StringSequenceToKShinglePipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceToKShinglePipe>(){}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<List<String>> context) {
        List<String> characters = buildCharList(context.getData());
        List<String> out = Lists.newArrayList();
        
        buff.setLength(0); 
        
        for(int i = 0; i < characters.size(); i++) {
            String token = characters.get(i).toString();
            for(int len : shingleSizes) {
                if(len > i+1) continue;
                if(len == 1){ out.add(token); continue; }
                buff.setLength(0);
                
                for(int k = len-1; k >= 1; k--)
                    buff.append(characters.get(i-k));
                buff.append(token);
                out.add(buff.toString());
            }
        }  
        return Context.createContext(context, out);
	}
}
