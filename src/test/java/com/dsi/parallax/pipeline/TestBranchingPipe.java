/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.pipeline;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Iterator;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.MultiPipe;
import com.dsi.parallax.pipeline.Pipe;
import com.dsi.parallax.pipeline.ReplacementPipe;
import com.dsi.parallax.pipeline.text.StringSequenceBranchingPipe;
import com.google.common.collect.Lists;

public class TestBranchingPipe {

	@Test
	public void testBranching() {
		List<String> a = Lists.newLinkedList();
		List<String> b = Lists.newLinkedList();
		List<String> c = Lists.newLinkedList();

		List<String> x = Lists.newLinkedList();

		a.add("a");
		b.add("b");
		c.add("c");
		x.add("x");

		@SuppressWarnings("unchecked")
		List<Context<List<String>>> comb = Lists.newArrayList(
				new Context<List<String>>(a), new Context<List<String>>(b),
				new Context<List<String>>(c));
		Iterator<Context<List<String>>> it = comb.iterator();

		StringSequenceBranchingPipe branching = new StringSequenceBranchingPipe();
		int branchfactor = 5;
		for (int i = 0; i < branchfactor; i++) {
			branching
					.addBranch(new ReplacementPipe<List<String>, List<String>>(
							x));
		}
		Iterator<Context<List<String>>> out = branching.processIterator(it);

		assertTrue(out.hasNext());
		int ct = 0;
		while (out.hasNext()) {
			List<String> list = out.next().getData();

			++ct;
			assertEquals(branchfactor, list.size());
			for (String v : list)
				assertEquals(v, "x");
		}
		assertEquals(ct, comb.size());
	}

	@Test
	public void testDifferentLengthBranching() {
		List<String> a = Lists.newArrayList("a");
		List<String> b = Lists.newArrayList("b");
		List<String> c = Lists.newArrayList("c");

		List<String> x = Lists.newArrayList("x");

		@SuppressWarnings("unchecked")
		List<Context<List<String>>> comb = Lists.newArrayList(
				new Context<List<String>>(a), new Context<List<String>>(b),
				new Context<List<String>>(c));
		Iterator<Context<List<String>>> it = comb.iterator();

		StringSequenceBranchingPipe branching = new StringSequenceBranchingPipe();
		int branchfactor = 5;
		for (int i = 0; i < branchfactor; i++) {
			branching
					.addBranch(new ReplacementPipe<List<String>, List<String>>(
							x));
		}
		MultiPipe<List<String>, List<String>> mp = MultiPipe
				.buildMultiPipe(
						new ReplacementPipe<List<String>, List<String>>(x))
				.addPipe(new ReplacementPipe<List<String>, List<String>>(x))
				.addPipe(new ReplacementPipe<List<String>, List<String>>(x));
		branching.addBranch(mp);

		Iterator<Context<List<String>>> out = branching.processIterator(it);

		assertTrue(out.hasNext());
		int ct = 0;
		while (out.hasNext()) {
			List<String> list = out.next().getData();
			++ct;
			assertEquals(branchfactor + 1, list.size());
			for (String v : list)
				assertEquals(v, "x");
		}
		assertEquals(ct, comb.size());
	}

	@Test
	public void testBranchAddition() {
		List<String> a = Lists.newLinkedList();
		List<String> b = Lists.newLinkedList();
		List<String> c = Lists.newLinkedList();

		List<String> x = Lists.newLinkedList();

		a.add("a");
		b.add("b");
		c.add("c");
		x.add("x");

		@SuppressWarnings("unchecked")
		List<Context<List<String>>> comb = Lists.newArrayList(
				new Context<List<String>>(a), new Context<List<String>>(b),
				new Context<List<String>>(c));
		Iterator<Context<List<String>>> it = comb.iterator();

		StringSequenceBranchingPipe branching = new StringSequenceBranchingPipe();
		int branchfactor = 5;
		for (int i = 0; i < branchfactor; i++) {
			branching
					.addBranch(new ReplacementPipe<List<String>, List<String>>(
							x));
		}
		Iterator<Context<List<String>>> out = branching.processIterator(it);

		assertTrue(out.hasNext());
		int ct = 0;
		while (out.hasNext()) {
			List<String> list = out.next().getData();

			assertEquals(branchfactor + ct, list.size());
			for (String v : list)
				assertEquals(v, "x");
			++ct;
			branching
					.addBranch(new ReplacementPipe<List<String>, List<String>>(
							x));
		}
		assertEquals(ct, comb.size());
	}

	@Test
	public void testBranchSubtraction() {
		List<String> a = Lists.newLinkedList();
		List<String> b = Lists.newLinkedList();
		List<String> c = Lists.newLinkedList();

		List<String> x = Lists.newLinkedList();

		a.add("a");
		b.add("b");
		c.add("c");
		x.add("x");

		@SuppressWarnings("unchecked")
		List<Context<List<String>>> comb = Lists.newArrayList(
				new Context<List<String>>(a), new Context<List<String>>(b),
				new Context<List<String>>(c));
		Iterator<Context<List<String>>> it = comb.iterator();

		StringSequenceBranchingPipe branching = new StringSequenceBranchingPipe();
		int branchfactor = 5;
		for (int i = 0; i < branchfactor; i++) {
			branching
					.addBranch(new ReplacementPipe<List<String>, List<String>>(
							x));
		}
		Iterator<Context<List<String>>> out = branching.processIterator(it);

		Pipe<List<String>, List<String>> addPipe = new ReplacementPipe<List<String>, List<String>>(
				x);
		assertTrue(out.hasNext());
		
		int ct = 0;
		while (out.hasNext()) {
			if(ct%2 == 0)
				branching.addBranch(addPipe);
			else
				branching.removeBranch(addPipe);
			
			List<String> list = out.next().getData();
			
			assertEquals(branchfactor + (ct%2 == 0 ? 1 : 0), list.size());
			
			for (String v : list)
				assertEquals(v, "x");
			++ct;
			
		}
		assertEquals(ct, comb.size());
	}
}
