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
import com.dsi.parallax.ml.target.BinaryTargetNumericParser;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.instance.BinaryInstancesFromVectorPipe;

/**
 * The Class TestBinaryInstancesFromVectorPipe.
 */
public class TestBinaryInstancesFromVectorPipe {

	/**
	 * Test instance same as vector.
	 */
	@Test
	public void testInstanceSameAsVector() {
		LinearVector vec = LinearVectorFactory.getVector(100);
		for (int i = 0; i < 100; i++)
			vec.resetValue(i, i * i);
		Context<LinearVector> context = new Context<LinearVector>(vec);
		BinaryInstancesFromVectorPipe pipe = new BinaryInstancesFromVectorPipe();
		Iterator<Context<BinaryClassificationInstance>> it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		assertTrue(it.hasNext());
		Context<BinaryClassificationInstance> c2 = it.next();
		BinaryClassificationInstance inst = c2.getData();
		for (int x : vec) {
			assertEquals(vec.getValue(x), inst.getFeatureValue(x), 0.00001);
		}
		for (int x : inst) {
			assertEquals(vec.getValue(x), inst.getFeatureValue(x), 0.00001);
		}
	}

	/**
	 * Test instance multiple label.
	 */
	@Test
	public void testInstanceMultipleLabel() {
		LinearVector vec = LinearVectorFactory.getVector(100);
		for (int i = 0; i < 100; i++)
			vec.resetValue(i, i * i);
		Context<LinearVector> context = new Context<LinearVector>(vec);

		BinaryInstancesFromVectorPipe pipe;
		pipe = new BinaryInstancesFromVectorPipe(
				new BinaryTargetNumericParser());
		Iterator<Context<BinaryClassificationInstance>> it;
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		assertTrue(it.hasNext());
		Context<BinaryClassificationInstance> c2;
		c2 = it.next();
		BinaryClassificationInstance inst;
		inst = c2.getData();
		for (int x : vec) {
			assertEquals(vec.getValue(x), inst.getFeatureValue(x), 0.00001);
		}
		for (int x : inst) {
			assertEquals(vec.getValue(x), inst.getFeatureValue(x), 0.00001);
		}
		context = new Context<LinearVector>(vec);
		context.setLabel("x");

		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null == inst.getLabel());

		context = new Context<LinearVector>(vec);
		context.setLabel(1 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null != inst.getLabel());

		context = new Context<LinearVector>(vec);
		context.setLabel(0 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null != inst.getLabel());

		pipe = new BinaryInstancesFromVectorPipe(
				new BinaryTargetNumericParser());

		context = new Context<LinearVector>(vec);
		context.setLabel("x");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null == inst.getLabel());

		context = new Context<LinearVector>(vec);
		context.setLabel(1 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null != inst.getLabel());
		assertEquals(inst.getLabel().getValue(), 1, 0.0000001);

		context = new Context<LinearVector>(vec);
		context.setLabel(0 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
	}

	/**
	 * Test instance with binary label.
	 */
	@Test
	public void testInstanceWithBinaryLabel() {
		LinearVector vec = LinearVectorFactory.getVector(100);
		for (int i = 0; i < 100; i++)
			vec.resetValue(i, i * i);
		Context<LinearVector> context = new Context<LinearVector>(vec);

		BinaryInstancesFromVectorPipe pipe;
		pipe = new BinaryInstancesFromVectorPipe(
				new BinaryTargetNumericParser());
		Iterator<Context<BinaryClassificationInstance>> it;
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		assertTrue(it.hasNext());
		Context<BinaryClassificationInstance> c2;
		c2 = it.next();
		BinaryClassificationInstance inst;
		inst = c2.getData();
		for (int x : vec) {
			assertEquals(vec.getValue(x), inst.getFeatureValue(x), 0.00001);
		}

		context = new Context<LinearVector>(vec);
		context.setLabel("x");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null == inst.getLabel());

		context = new Context<LinearVector>(vec);
		context.setLabel(1 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null != inst.getLabel());
		assertEquals(inst.getLabel().getValue(), 1, 0.0000001);

		context = new Context<LinearVector>(vec);
		context.setLabel(0 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null != inst.getLabel());
		assertEquals(inst.getLabel().getValue(), 0, 0.0000001);

		context = new Context<LinearVector>(vec);
		context.setLabel(1000 + "");
		it = pipe
				.processIterator(Collections.singletonList(context).iterator());
		c2 = it.next();
		inst = c2.getData();
		assertTrue(null == inst.getLabel());

	}

}
