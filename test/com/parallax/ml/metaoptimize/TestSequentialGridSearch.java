/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.metaoptimize;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.google.common.collect.Sets;
import com.parallax.ml.classifier.Classifiers;
import com.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PerceptronWithMarginBuilder;
import com.parallax.ml.util.option.Configuration;

/**
 * The Class TestSequentialGridSearch.
 */
public class TestSequentialGridSearch {

	/** The classifiers. */
	Classifiers classifiers = Classifiers.MARGINPERCEPTRON;
	
	/**
	 * Test gives different configs.
	 */
	@Test
	public void testGivesDifferentConfigs() {
		@SuppressWarnings("unchecked")
		Configuration<PerceptronWithMarginBuilder> config = (Configuration<PerceptronWithMarginBuilder>)Classifiers.MARGINPERCEPTRON.getConfiguration();
		SequentialGridSearch<PerceptronWithMarginBuilder> sgs = new SequentialGridSearch<PerceptronWithMarginBuilder>(config, 10);

		String[] last = config.getArgumentsFromOpts();
		sgs.next();
		while (sgs.hasNext()) {
			String[] next = sgs.next().getArgumentsFromOpts();
			assertTrue(!Sets.newHashSet(last).equals(Sets.newHashSet(next)));
			last = next;
		}
	}

	/**
	 * Specify optimized options.
	 */
	@Test
	public void specifyOptimizedOptions() {
		@SuppressWarnings("unchecked")
		Configuration<PerceptronWithMarginBuilder> config = (Configuration<PerceptronWithMarginBuilder>)Classifiers.MARGINPERCEPTRON.getConfiguration();
		SequentialGridSearch<PerceptronWithMarginBuilder> sgs = new SequentialGridSearch<PerceptronWithMarginBuilder>(config, new String[] {}, 10);

		String[] last = config.getArgumentsFromOpts();

		int ct = 0;
		while (sgs.hasNext() && ct++ < 100) {
			String[] next = sgs.next().getArgumentsFromOpts();
			assertTrue(Sets.newHashSet(last).equals(Sets.newHashSet(next)));
			last = next;
		}

		sgs = new SequentialGridSearch<PerceptronWithMarginBuilder>(config, new String[] { "GR" }, 20);
		double lastGR = config.floatOptionFromShortName("GR");
		sgs.next();
		while (sgs.hasNext()) {
			double nextGR = sgs.next().floatOptionFromShortName("GR");
			assertTrue(nextGR != lastGR);
			lastGR = nextGR;
		}
	}


}
