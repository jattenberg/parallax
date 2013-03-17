package com.parallax.optimization.stochastic.anneal;

import com.parallax.ml.util.option.NestedConfigurable;

public abstract class AnnealingScheduleBuilder extends NestedConfigurable<AnnealingScheduleBuilder, AnnealingScheduleType> {

	private static final long serialVersionUID = 6678552487106653096L;
	
	public abstract AnnealingSchedule build();
}
