/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.pipeline;

import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.google.common.collect.Lists;

public class TestFoldFilterPipe {

    @Test
    public void test() {
        List<Context<Integer>> terms = Lists.newArrayList();
        for(int i = 0; i < 100; i++) {
            terms.add(new Context<Integer>(i));
        }
        int folds = 10;
        for(int fold = 0; fold < folds; fold ++) {
            FoldFilterPipe<Integer> pipe = new FoldFilterPipe<Integer>(fold, folds);
            Iterator<Context<Integer>> out = pipe.processIterator(terms.iterator());
            assertTrue(out.hasNext());
            while(out.hasNext()) {
                Context<Integer> con = out.next();
                int val = con.getData();
                assertTrue(val%folds == fold);
            }
            assertTrue(!out.hasNext());
        }            
            
    }

}
