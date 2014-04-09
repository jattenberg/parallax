package com.dsi.parallax.optimization.stochastic.anneal;

import com.dsi.parallax.optimization.Gradient;

public interface GradientUpdateable {

	public abstract void update(Gradient grad);
}
