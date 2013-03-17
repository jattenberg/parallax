/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.parallax.ml.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ResivoirSampler<T>
{
	List<T> pool;
	int get;
	Random rand = new Random(System.currentTimeMillis());
	int index;
	
	public ResivoirSampler(int get)
	{
		this.get = get;
		pool = new ArrayList<T>();
		this.index = 0;
	}
	public void observe(T o)
	{
		if(index < get)
			pool.add(o);
		else
		{
			int id = rand.nextInt(index);
			if(id<get)
				pool.set(id, o);
		}
		index++;
	}
	public List<T> getPool()
	{
		return pool;
	}
}
