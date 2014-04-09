/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import java.io.Serializable;
import java.util.Collection;


public interface Discretization extends Serializable {
    
    public void build(Collection<Double> data);
    public double descretize(double datum);
    public boolean isBuilt();
}
