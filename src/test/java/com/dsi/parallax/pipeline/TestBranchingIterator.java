/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.junit.Test;

import com.dsi.parallax.pipeline.BranchingIteratorMaker;
import com.dsi.parallax.pipeline.Context;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

public class TestBranchingIterator {

	@Test
	public void testAdvancesBrances() {
		List<String> a = Lists.newArrayList("a");
		List<String> b = Lists.newArrayList("b");
		List<String> c = Lists.newArrayList("c");

		@SuppressWarnings("unchecked")
		List<Context<List<String>>> comb = Lists.newArrayList(
				new Context<List<String>>(a), new Context<List<String>>(b),
				new Context<List<String>>(c));
		Iterator<Context<List<String>>> it = comb.iterator();
		assertTrue(it.hasNext());

		BranchingIteratorMaker<List<String>> bIterator = new BranchingIteratorMaker<List<String>>(
				it);

		Set<Iterator<Context<List<String>>>> bits = Sets.newHashSet();

		for (int i = 0; i < 5; i++) {
			Iterator<Context<List<String>>> bit = bIterator
					.buildBranchIterator();
			bits.add(bit);
		}
		for (Iterator<Context<List<String>>> bit : bits) {
			assertTrue(bit.hasNext());

			Context<List<String>> next;
			next = bit.next();

			assertEquals(next.getData().get(0), "a");

			next = bit.next();

			assertEquals(next.getData().get(0), "b");

			next = bit.next();

			assertEquals(next.getData().get(0), "c");

			assertFalse(bit.hasNext());
		}
		assertFalse(bIterator.buildBranchIterator().hasNext());
	}

	@Test
	public void testRestarts() {
		String a = "a", b = "b", c = "c";

		@SuppressWarnings("unchecked")
		List<Context<String>> comb = Lists.newArrayList(new Context<String>(a),
				new Context<String>(b), new Context<String>(c));
		Iterator<Context<String>> it = comb.iterator();
		assertTrue(it.hasNext());

		BranchingIteratorMaker<String> bIterator = new BranchingIteratorMaker<String>(
				it);
		Iterator<Context<String>> bit = bIterator.buildBranchIterator();
		assertTrue(bit.hasNext());
		assertEquals(bit.next().getData(), "a");
		Iterator<Context<String>> bit2 = bIterator.buildBranchIterator();
		assertTrue(bit.hasNext());
		assertEquals(bit.next().getData(), "b");
		
		assertTrue(bit2.hasNext());
		assertEquals(bit2.next().getData(), "b");
		
		Iterator<Context<String>> bit3 = bIterator.buildBranchIterator();
		assertTrue(bit.hasNext());
		assertEquals(bit.next().getData(), "c");
		assertTrue(bit2.hasNext());
		assertEquals(bit2.next().getData(), "c");
		assertTrue(bit3.hasNext());
		assertEquals(bit3.next().getData(), "c");
		
		Iterator<Context<String>> bit4 = bIterator.buildBranchIterator();
		assertFalse(bit.hasNext());
		assertFalse(bit2.hasNext());
		assertFalse(bit3.hasNext());
		assertFalse(bit4.hasNext());
	}
	
	@Test
	public void testRemoves() {
		String a = "a", b = "b", c = "c";

		@SuppressWarnings("unchecked")
		List<Context<String>> comb = Lists.newArrayList(new Context<String>(a),
				new Context<String>(b), new Context<String>(c));
		Iterator<Context<String>> it = comb.iterator();
		assertTrue(it.hasNext());

		BranchingIteratorMaker<String> bIterator = new BranchingIteratorMaker<String>(
				it);
		Iterator<Context<String>> bit = bIterator.buildBranchIterator();
		assertTrue(bit.hasNext());
		assertEquals(bit.next().getData(), "a");
		bIterator.remove(bit);
		
		Iterator<Context<String>> bit2 = bIterator.buildBranchIterator();
		
		assertTrue(bit2.hasNext());
		assertEquals(bit2.next().getData(), "b");
		bIterator.remove(bit2);
		
		Iterator<Context<String>> bit3 = bIterator.buildBranchIterator();

		assertTrue(bit3.hasNext());
		assertEquals(bit3.next().getData(), "c");
		
		bIterator.remove(bit3);
		
		Iterator<Context<String>> bit4 = bIterator.buildBranchIterator();
		assertFalse(bit4.hasNext());
	}

	@Test
	public void testRemovesParallel() {
		String a = "a", b = "b", c = "c";

		@SuppressWarnings("unchecked")
		List<Context<String>> comb = Lists.newArrayList(new Context<String>(a),
				new Context<String>(b), new Context<String>(c));
		Iterator<Context<String>> it = comb.iterator();
		assertTrue(it.hasNext());

		BranchingIteratorMaker<String> bIterator = new BranchingIteratorMaker<String>(
				it);
		Iterator<Context<String>> bit = bIterator.buildBranchIterator();
		Iterator<Context<String>> bit2 = bIterator.buildBranchIterator();
		assertTrue(bit.hasNext());
		assertEquals(bit.next().getData(), "a");
		bIterator.remove(bit);
		
		assertTrue(bit2.hasNext());
		assertEquals(bit2.next().getData(), "a");
		bIterator.remove(bit2);
		
		Iterator<Context<String>> bit3 = bIterator.buildBranchIterator();

		assertTrue(bit3.hasNext());
		assertEquals(bit3.next().getData(), "b");
	}
}
