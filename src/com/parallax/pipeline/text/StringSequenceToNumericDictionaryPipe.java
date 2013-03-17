/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline.text;

import java.lang.reflect.Type;
import java.util.List;

import com.google.gson.reflect.TypeToken;
import com.parallax.ml.dictionary.Dictionary;
import com.parallax.ml.dictionary.HashDictionary;
import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.AbstractPipe;
import com.parallax.pipeline.Context;

/**
 * StringSequenceToNumericDictionaryPipe is primitive class for getting List contents of String object into context
 *
 * @author Josh Attenberg
 */
public class StringSequenceToNumericDictionaryPipe extends AbstractPipe<List<String>, LinearVector> {

    private static final long serialVersionUID = -6000453314033083461L;
    private Dictionary dict;

    /**
     * Class constructor specifying size to create
     * @param size size
     */
    public StringSequenceToNumericDictionaryPipe(int size) {
        this(new HashDictionary(size));
    }

    /**
     * Class constructor specifying Dictionary to create
     * @param dict Dictionary
     */
    public StringSequenceToNumericDictionaryPipe(Dictionary dict) {
    	super();
        this.dict = dict;      
    }

    /**
     * The method returns the class's Type "StringSequenceToNumericDictionaryPipe"
     * @return  Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceToNumericDictionaryPipe>(){}.getType();
	}

	@Override
	protected Context<LinearVector> operate(Context<List<String>> context) {
        List<String> tokens = context.getData();
        LinearVector vector = dict.vectorFromText(tokens);
        return Context.createContext(context, vector);
	}


}
