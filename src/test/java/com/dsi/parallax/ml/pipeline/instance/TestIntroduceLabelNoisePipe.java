/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.ml.pipeline.instance;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Collections;
import java.util.Iterator;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.target.BinaryClassificationTarget;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.instance.IntroduceLabelNoisePipe;


/**
 * The Class TestIntroduceLabelNoisePipe.
 */
public class TestIntroduceLabelNoisePipe {

	/**
	 * Test pos.
	 */
	@Test
	public void testPos() {
		for(double r = 0; r <= 1; r+=0.2) {
			
			double tot = 0;
			IntroduceLabelNoisePipe pipe = new IntroduceLabelNoisePipe(r);
			
			for(int i = 0; i < 10000; i++) {
				BinaryClassificationInstance inst = new BinaryClassificationInstance(5);
				inst.setLabel(new BinaryClassificationTarget(1));
				Context<BinaryClassificationInstance> in = new Context<BinaryClassificationInstance>(inst);
				Context<BinaryClassificationInstance> out;
				Iterator<Context<BinaryClassificationInstance>> it = pipe.processIterator(Collections.singletonList(in).iterator());
				
				assertTrue(it.hasNext());
				out = it.next();
				tot += out.getData().getLabel().getValue();
			}
			double ratio = tot/10000.0;
			assertEquals(1.-r, ratio, 0.1);
		}
	}

	
	/**
	 * Test neg.
	 */
	@Test
	public void testNeg() {
		for(double r = 0; r <= 1; r+=0.2) {
			
			double tot = 0;
			IntroduceLabelNoisePipe pipe = new IntroduceLabelNoisePipe(r);
			
			for(int i = 0; i < 10000; i++) {
				BinaryClassificationInstance inst = new BinaryClassificationInstance(5);
				inst.setLabel(new BinaryClassificationTarget(0));
				Context<BinaryClassificationInstance> in = new Context<BinaryClassificationInstance>(inst);
				Context<BinaryClassificationInstance> out;
				Iterator<Context<BinaryClassificationInstance>> it = pipe.processIterator(Collections.singletonList(in).iterator());
				
				assertTrue(it.hasNext());
				out = it.next();
				tot += out.getData().getLabel().getValue();
			}
			double ratio = tot/10000.0;
			assertEquals(r, ratio, 0.1);
		}
	}
}
