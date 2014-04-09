/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.ReplacementPipe;
import com.google.common.collect.Lists;

public class TestReplacementPipe {

    @Test
    public void testReplacement() {
        List<String> a = Lists.newArrayList("a");
        List<String> b = Lists.newArrayList("b");
        List<String> c = Lists.newArrayList("c");
        
        List<String> x = Lists.newArrayList("x");

        @SuppressWarnings("unchecked")
        List<Context<List<String>>> comb = Lists.newArrayList(new Context<List<String>>(a),new Context<List<String>>(b),new Context<List<String>>(c));
        Iterator<Context<List<String>>> it = comb.iterator();
        
        ReplacementPipe<List<String>,List<String>> pipe = new  ReplacementPipe<List<String>,List<String>>(x);
        Iterator<Context<List<String>>> out = pipe.processIterator(it);
        
        assertTrue(out.hasNext());
        int ct = 0;
        while(out.hasNext()) {
            List<String> list = out.next().getData();
            ++ct;
            assertEquals(list.get(0),"x");
        }
        assertEquals(ct,comb.size());

    }
}
