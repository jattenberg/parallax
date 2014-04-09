/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.StringSequenceStopWordsFilterPipe;
import com.google.common.collect.Lists;

public class TestStringSequenceStopWordsFilterPipe {
    /**
     * This method is to set  caseSensitive = false, make the test data convert to lower case
     */
    @Test
    public void test()
    {
        //Create test data
        List<String> a = Lists.newArrayList("A");
        List<String> b = Lists.newArrayList("B");
        List<String> c = Lists.newArrayList("C");
        //Create test data for stop list
        String[] stopWords = {"a","b","d"};

        @SuppressWarnings("unchecked")
        List<Context<List<String>>> comb = Lists.newArrayList(new Context<List<String>>(a),
                new Context<List<String>>(b),new Context<List<String>>(c));

        Iterator<Context<List<String>>> it = comb.iterator();
        //Set parameter is false to make test data convert to Lower case
        StringSequenceStopWordsFilterPipe pipe = new StringSequenceStopWordsFilterPipe(stopWords,false);
        assertTrue(getCountOfActualData(pipe,it) < comb.size());
    }

    /**
     * This method is to set  caseSensitive = true, make the test data not convert
     */
    @Test
    public void testInvalid()
    {
        //Create test data
        List<String> a = Lists.newArrayList("A");
        List<String> b = Lists.newArrayList("B");
        List<String> c = Lists.newArrayList("C");
        //Create test data for stop list
        String[] stopWords = {"a","b","d"};

        @SuppressWarnings("unchecked")
        List<Context<List<String>>> comb = Lists.newArrayList(new Context<List<String>>(a),
                new Context<List<String>>(b),new Context<List<String>>(c));

        Iterator<Context<List<String>>> it = comb.iterator();
        //Set parameter is true to use test data and not convert
        StringSequenceStopWordsFilterPipe pipe = new StringSequenceStopWordsFilterPipe(stopWords,true);
        assertTrue(getCountOfActualData(pipe,it) == comb.size());
    }

    /**
     * This method is to get the count of executing data
     * @param pipe StringSequenceStopWordsFilterPipe
     * @param it Iterator<Context<List<String>>>
     * @return int the count of executing
     */
    private int getCountOfActualData(StringSequenceStopWordsFilterPipe pipe,Iterator<Context<List<String>>> it)
    {
        Iterator<Context<List<String>>> contextIterator = pipe.processIterator(it);
        int ct = 0;
        while(contextIterator.hasNext()) {
            List<String> test = contextIterator.next().getData();
            if(test!=null&&test.size()!=0)
            {
                ++ct;
            }
        }
        return ct;
    }
}
