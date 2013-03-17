package com.parallax.optimization.stochastic.anneal;

import com.parallax.optimization.Gradient;

public interface GradientUpdateable {

	public abstract void update(Gradient grad);
}
