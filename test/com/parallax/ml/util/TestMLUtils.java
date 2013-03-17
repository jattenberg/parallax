/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.Random;

import org.junit.Test;

public class TestMLUtils
{
	private static final Random gen = new Random();

	@Test
	public void testQuickSort()
	{
		for(int i = 0; i < 100; i++)
		{
			Double[] d = new Double[100];
			Long[] L = new Long[100];
			Float[] f = new Float[100];
			Integer[] I = new Integer[100];
			for(int j = 0; j < 100; j++)
			{
				d[j]=gen.nextDouble();
				L[j]=gen.nextLong();
				f[j]=gen.nextFloat();
				I[j]=gen.nextInt();
			}
			MLUtils.quicksort(d);
			MLUtils.quicksort(L);
			MLUtils.quicksort(f);
			MLUtils.quicksort(I);
			assertTrue(isSorted(d));
			assertTrue(isSorted(L));
			assertTrue(isSorted(f));
			assertTrue(isSorted(I));
		}
	}
	@Test
	public void testSelect()
	{
		for(int i = 0; i < 100; i++)
		{
			Double[] d = new Double[100];
			Long[] L = new Long[100];
			Float[] f = new Float[100];
			Integer[] I = new Integer[100];
			for(int j = 0; j < 100; j++)
			{
				d[j]=gen.nextDouble();
				L[j]=gen.nextLong();
				f[j]=gen.nextFloat();
				I[j]=gen.nextInt();
			}
			Double[] dc = new Double[100];
			System.arraycopy(d, 0, dc, 0, 100);
			Long[] Lc = new Long[100];
			System.arraycopy(L, 0, Lc, 0, 100);
			Float[] fc = new Float[100];
			System.arraycopy(f, 0, fc, 0, 100);
			Integer[] Ic = new Integer[100];
			System.arraycopy(I, 0, Ic, 0, 100);
			
			MLUtils.quicksort(d);
			MLUtils.quicksort(L);
			MLUtils.quicksort(f);
			MLUtils.quicksort(I);
			for(int j = 0; j < 100; j++)
			{
				assertEquals(d[j], MLUtils.slowselect(dc, j), 0.000000001);
				assertEquals(L[j], MLUtils.slowselect(Lc, j), 0.000000001);
				assertEquals(f[j], MLUtils.slowselect(fc, j), 0.000000001);
				assertEquals(I[j], MLUtils.slowselect(Ic, j), 0.000000001);
			}
			for(int j = 0; j < 100; j++)
			{
				assertEquals(d[j], MLUtils.quickselect(dc, j), 0.000000001);
				assertEquals(L[j], MLUtils.quickselect(Lc, j), 0.000000001);
				assertEquals(f[j], MLUtils.quickselect(fc, j), 0.000000001);
				assertEquals(I[j], MLUtils.quickselect(Ic, j), 0.000000001);
			}
		}
	}

	private <T extends Comparable<T>> boolean isSorted(T[]  d)
	{
		if(d.length<=1)
			return true;
		T last = d[0];
		for(int i = 1; i < d.length; i++)
		{
			if(last.compareTo(d[i])>0)
				return false;
			last = d[i];
		}
		return true;
	}
	
	@Test
	public void testLogAdd() {

		for(int i = 1; i <= 100; i+=5) 
			for(int j = 0; j <= 100; j+=5)
				assertEquals(MLUtils.logAdd(Math.log(i), Math.log(j)), Math.log(i+j), 0.00001);
	}
}
