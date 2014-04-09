/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import org.junit.Test;

import com.dsi.parallax.ml.util.Histogram;
import com.dsi.parallax.ml.util.MLUtils;

public class TestHistogram {

    @Test
    public void test() {
        Histogram hist = new Histogram();
        hist.allocate(4);
        for(int i = 0; i < 50000; i++) {
            hist.add(MLUtils.GENERATOR.nextInt(5));
        }
    }

}
