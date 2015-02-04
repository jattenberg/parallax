/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.pipeline.AbstractBranchingPipe;
import com.dsi.parallax.pipeline.StringSequenceConcatingCombiner;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.List;

/**
 * yes, this is the whole class. 
 * OOP ftw.
 * @author jattenberg
 *
 */
public class StringSequenceBranchingPipe extends AbstractBranchingPipe<List<String>, List<String>, List<String>>{
    
    private static final long serialVersionUID = 3338689311414230217L;

    /**
     * Class constructor
     */
    public StringSequenceBranchingPipe() {
        super();
        addCombiner(new StringSequenceConcatingCombiner());
    }

    /**
     * The method returns the class's Type "StringSequenceBranchingPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceBranchingPipe>(){}.getType();
	}

}
