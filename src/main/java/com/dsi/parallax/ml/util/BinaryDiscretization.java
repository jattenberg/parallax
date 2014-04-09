/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import java.util.Collection;

public class BinaryDiscretization implements Discretization {

    private static final long serialVersionUID = 3285830377677502105L;

    @Override
    public void build(Collection<Double> data) {
        //pass
    }

    @Override
    public double descretize(double datum) {
        return Math.abs(datum) > 0 ? 1 : 0;
    }

    @Override
    public boolean isBuilt() {
        return true;
    }
}
