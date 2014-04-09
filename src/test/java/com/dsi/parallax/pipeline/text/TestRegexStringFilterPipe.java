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
import com.dsi.parallax.pipeline.text.RegexStringFilterPipe;
import com.google.common.collect.Lists;

public class TestRegexStringFilterPipe {

	  private static final String REGEX_VALUE = "\\?"; // Regex value for test data

	    @Test
	    public void test()
	    {
	        String a = "a",b = "b", c = "c", q = "?", q2 = "?awerawe";

	        @SuppressWarnings("unchecked")
	        List<Context<String>> comb = Lists.newArrayList(new Context<String>(a),
	                new Context<String>(b),new Context<String>(c), new Context<String>(q), new Context<String>(q2));

	        Iterator<Context<String>> it = comb.iterator();

	        RegexStringFilterPipe pipe = new RegexStringFilterPipe(REGEX_VALUE);
	        Iterator<Context<String>> contextIterator = pipe.processIterator(it);

	        assertTrue(contextIterator.hasNext());
	        int ct = 0;
	        while(contextIterator.hasNext()) {
	            String str = contextIterator.next().getData();
	            ++ct;
	            assertTrue(!str.contains(REGEX_VALUE));
	        }
	        assertEquals(ct,comb.size()-2);
	    }

}
