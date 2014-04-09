package com.dsi.parallax.optimization.stochastic.anneal;

import com.dsi.parallax.ml.util.option.NestedConfigurableOption;

public class AnnealingScheduleConfigurableOption extends NestedConfigurableOption<AnnealingScheduleType, AnnealingScheduleBuilder, AnnealingScheduleConfigurableBuilder> {

	private static final long serialVersionUID = -8953188482690371956L;
	
	public AnnealingScheduleConfigurableOption(
			String shortName,
			String longName,
			boolean optimizable,
			String desc) {
		super(shortName, longName, optimizable, desc, AnnealingScheduleConfigurableBuilder.options);
	}
	
}