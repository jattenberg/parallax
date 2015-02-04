/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import com.dsi.parallax.ml.util.StopWordSet;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;
import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * pipe for removing stopwords
 *
 * @author jattenberg
 */
public class StringSequenceStopWordsFilterPipe extends AbstractPipe<List<String>,List<String>> {

    private static final long serialVersionUID = -1097762044581412929L;
    private boolean caseSensitive = false;

    private Set<String> stoplist;

    /**
     * Class constructor.
     */
    public StringSequenceStopWordsFilterPipe() {
    	super();
    	stoplist = StopWordSet.stopwords;
    }

    /**
     * Class constructor specifying caseSensitive boolean of objects to create
     * @param caseSensitive case sensitive
     */
    public StringSequenceStopWordsFilterPipe(boolean caseSensitive) {
        this();
        this.caseSensitive = caseSensitive;
    }

    /**
     * Class constructor specifying multiple stop words to create
     * @param altStopWords multiple stop words
     */
    public StringSequenceStopWordsFilterPipe(String[] altStopWords ) {
        super();
    	stoplist = Sets.newHashSet(altStopWords);
    }

    /**
     * Class constructor specifying multiple altStopWords to create
     * @param altStopWords multiple stop words
     */
    public StringSequenceStopWordsFilterPipe(Collection<String> altStopWords ) {
        super();
    	stoplist = Sets.newHashSet(altStopWords);
    }

    /**
     * Class constructor specifying multiple stop words and caseSensitive boolean to create
     * @param altStopWords multiple stop words
     * @param caseSensitive case sensitive
     */
    public StringSequenceStopWordsFilterPipe(String[] altStopWords, boolean caseSensitive) {
        this(altStopWords);
        this.caseSensitive = caseSensitive;
    }

    /**
     * Class constructor specifying multiple stop words and caseSensitive boolean to create
     * @param altStopWords multiple stop words
     * @param caseSensitive case sensitive
     */
    public StringSequenceStopWordsFilterPipe(Collection<String> altStopWords, boolean caseSensitive ) {
        this(altStopWords);
        this.caseSensitive = caseSensitive;
    }

    /**
     * The method returns the class's Type "StringSequenceStopWordsFilterPipe"
     * @return Type
     */
	@Override
	public Type getType() {
		return new TypeToken<StringSequenceStopWordsFilterPipe>(){}.getType();
	}

	@Override
	protected Context<List<String>> operate(Context<List<String>> context) {
        List<String> payload = context.getData();
        List<String> out = Lists.newLinkedList();
        
        for(String term : payload) {
            if(!stoplist.contains( caseSensitive ? term : term.toLowerCase() ))
                out.add(term);
        }
        return Context.createContext(context, out);
	}
    
}
