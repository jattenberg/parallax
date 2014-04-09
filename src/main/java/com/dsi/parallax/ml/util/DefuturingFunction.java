/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.util;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;

import com.google.common.base.Function;

public class DefuturingFunction<F> implements Function<Future<F>, F> {

	@Override
	public F apply(Future<F> future) {
		try {
			return future.get();
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		} catch (ExecutionException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static <F> DefuturingFunction<F> createDefuturingFunction() {
		return new DefuturingFunction<F>();
	}

}
