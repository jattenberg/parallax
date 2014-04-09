/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.metaoptimize;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.classifier.Classifiers;
import com.dsi.parallax.ml.classifier.linear.updateable.LinearUpdateableClassifierBuilder.PerceptronWithMarginBuilder;
import com.dsi.parallax.ml.metaoptimize.RandomGridSearch;
import com.dsi.parallax.ml.util.option.Configuration;
import com.google.common.collect.Sets;

/**
 * The Class TestRandomGridSearch.
 */
public class TestRandomGridSearch {

	/** The classifiers. */
	Classifiers classifiers = Classifiers.MARGINPERCEPTRON;

	/**
	 * Test gives random configs.
	 */
	@Test
	public void testGivesRandomConfigs() {
		@SuppressWarnings("unchecked")
		Configuration<PerceptronWithMarginBuilder> config = (Configuration<PerceptronWithMarginBuilder>) classifiers
				.getConfiguration();
		RandomGridSearch<PerceptronWithMarginBuilder> rgs = new RandomGridSearch<PerceptronWithMarginBuilder>(
				config);
		String[] last = config.getArgumentsFromOpts();

		int ct = 0;
		while (rgs.hasNext() && ct++ < 100) {
			String[] next = rgs.next().getArgumentsFromOpts();
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
		Configuration<PerceptronWithMarginBuilder> config = (Configuration<PerceptronWithMarginBuilder>) classifiers
				.getConfiguration();
		RandomGridSearch<PerceptronWithMarginBuilder> rgs = new RandomGridSearch<PerceptronWithMarginBuilder>(
				config, new String[] {});

		String[] last = config.getArgumentsFromOpts();

		int ct = 0;
		while (rgs.hasNext() && ct++ < 100) {
			String[] next = rgs.next().getArgumentsFromOpts();
			assertTrue(Sets.newHashSet(last).equals(Sets.newHashSet(next)));
			last = next;
		}

		rgs = new RandomGridSearch<PerceptronWithMarginBuilder>(config,
				new String[] { "GR" });
		double lastGR = config.floatOptionFromShortName("GR");
		ct = 0;
		while (rgs.hasNext() && ct++ < 300) {
			double nextGR = rgs.next().floatOptionFromShortName("GR");
			assertTrue(nextGR != lastGR);
			lastGR = nextGR;
		}
	}

}
