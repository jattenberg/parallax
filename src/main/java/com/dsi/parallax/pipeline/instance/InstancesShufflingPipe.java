package com.dsi.parallax.pipeline.instance;

import com.dsi.parallax.ml.instance.Instances;
import com.dsi.parallax.pipeline.AbstractPipe;
import com.dsi.parallax.pipeline.Context;
import com.google.gson.reflect.TypeToken;

import java.lang.reflect.Type;

public class InstancesShufflingPipe<I extends Instances<?>> extends AbstractPipe<I, I> {


	private static final long serialVersionUID = -3381580964526680831L;

	public InstancesShufflingPipe() {
		super();
	}

	@Override
	public Type getType() {
		return new TypeToken<InstancesShufflingPipe<I>>(){}.getType(); 
	}

	@Override
	protected Context<I> operate(Context<I> context) {
		@SuppressWarnings("unchecked")
		I instances = (I)context.getData().shuffle();
		return Context.createContext(context, instances);
	}

}
