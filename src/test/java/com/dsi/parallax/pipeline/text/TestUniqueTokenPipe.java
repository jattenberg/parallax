/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline.text;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.UniqueTokenPipe;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;

public class TestUniqueTokenPipe {

    List<String> tokens = Lists.newArrayList("foo", "bar", "baz");
    List<String> tokens2 = Lists.newArrayList("foo", "bar", "baz", "foo", "foo");
	
    
    @Test
	public void testNoUnnecessary() {
        for(int i = 1; i < 5; i++) {
        	Context<List<String>> context = new Context<List<String>>(tokens);
            UniqueTokenPipe pipe = new UniqueTokenPipe();
            @SuppressWarnings("unchecked")
			Iterator<Context<List<String>>> it = pipe.processIterator(Iterators.forArray(context));
            assertTrue(it.hasNext());
            List<String> vec = it.next().getData();
            assertTrue(!it.hasNext());
            assertEquals(vec.size(),tokens.size());
        }
	}
    
    @Test
	public void testUniques() {
        for(int i = 1; i < 5; i++) {
        	Context<List<String>> context = new Context<List<String>>(tokens2);
            UniqueTokenPipe pipe = new UniqueTokenPipe();
            @SuppressWarnings("unchecked")
			Iterator<Context<List<String>>> it = pipe.processIterator(Iterators.forArray(context));
            assertTrue(it.hasNext());
            List<String> vec = it.next().getData();
            assertTrue(!it.hasNext());
            assertEquals(vec.size(),tokens.size());
        }
    }
}
