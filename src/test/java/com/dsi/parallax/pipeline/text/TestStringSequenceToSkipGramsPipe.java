/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.StringSequenceToSkipGramsPipe;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;

public class TestStringSequenceToSkipGramsPipe {

    List<String> longTokens = Lists.newArrayList(MLUtils.randomString(100));
    List<String> tokens2 = Lists.newArrayList("foo", "bar", "baz");
    @Test
    public void test() {
        
        for(int i = 1; i < 5; i++) {
            Context<List<String>> context = new Context<List<String>>(tokens2);
            StringSequenceToSkipGramsPipe pipe = new StringSequenceToSkipGramsPipe(
                    new int[]{i});
            @SuppressWarnings("unchecked")
			Iterator<Context<List<String>>> it = pipe.processIterator(Iterators.forArray(context));
            assertTrue(it.hasNext());
            List<String> vec = it.next().getData();
            assertTrue(!it.hasNext());
            assertEquals(vec.size(),(int)Math.max(0, tokens2.size()-(i-1)));
        }
        
        for(int i = 1; i < 105; i++) {
            Context<List<String>> context = new Context<List<String>>(longTokens);
            StringSequenceToSkipGramsPipe pipe = new StringSequenceToSkipGramsPipe(
                    new int[]{i});
            @SuppressWarnings("unchecked")
			Iterator<Context<List<String>>> it = pipe.processIterator(Iterators.forArray(context));
            assertTrue(it.hasNext());
            List<String> vec = it.next().getData();
            assertTrue(!it.hasNext());
            assertEquals(vec.size(),(int)Math.max(0, longTokens.size()-(i-1)));
        }
    }


}
