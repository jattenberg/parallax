/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.google.common.collect.Lists;

public class TestMultiPipe {

    @Test
    public void test() {
        List<String> a = Lists.newArrayList("a");
        List<String> b = Lists.newArrayList("b");
        List<String> c = Lists.newArrayList("c");

        List<String> x = Lists.newArrayList("x");
        List<String> y = Lists.newArrayList("y");

        @SuppressWarnings("unchecked")
        List<Context<List<String>>> comb = Lists.newArrayList(
                new Context<List<String>>(a), new Context<List<String>>(b),
                new Context<List<String>>(c));
        Iterator<Context<List<String>>> it = comb.iterator();

        ReplacementPipe<List<String>, List<String>> xpipe = new ReplacementPipe<List<String>, List<String>>(
                x);
        ReplacementPipe<List<String>, List<String>> ypipe = new ReplacementPipe<List<String>, List<String>>(
                y);
        List<String> list;
        MultiPipe<List<String>, List<String>> mp = MultiPipe
                .buildMultiPipe(xpipe).addPipe(xpipe).addPipe(xpipe);

        Iterator<Context<List<String>>> out = mp.processIterator(it);

        assertTrue(out.hasNext());
        list = out.next().getData();
        for (String v : list)
            assertEquals(v, "x");
        assertEquals(list.size(), 1);

        
        MultiPipe<List<String>, List<String>> mp2 = MultiPipe.buildMultiPipe(xpipe).addPipe(ypipe);

        Iterator<Context<List<String>>> out2 = mp2.processIterator(it);

        assertTrue(out2.hasNext());
        list = out2.next().getData();

        for (String v : list)
            assertEquals(v, "y");
        assertEquals(list.size(), 1);

    }

}
