package com.dsi.parallax.optimization.stochastic.anneal;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleBuilder;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingScheduleType;
import com.dsi.parallax.optimization.stochastic.anneal.ConstantAnnealingSchedule;
import com.dsi.parallax.optimization.stochastic.anneal.ConstantAnnealingScheduleBuilder;

public class TestConstantAnnealingScheduleBuilder {

	@Test
	public void testCorrectType() {
		ConstantAnnealingScheduleBuilder builder = new ConstantAnnealingScheduleBuilder();
		assertEquals(builder.correspondingType(),
				AnnealingScheduleType.CONSTANT);
	}

	@Test
	public void testConfiguresInitialRate() {

		ConstantAnnealingScheduleBuilder builder = new ConstantAnnealingScheduleBuilder();

		builder.setInitialRate(5);
		assertEquals(builder.getLearningRate(), 5, 0.);
		ConstantAnnealingSchedule schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 5, 0.);

		Configuration<AnnealingScheduleBuilder> configuration = new Configuration<AnnealingScheduleBuilder>(
				ConstantAnnealingScheduleBuilder.options);
		configuration.addFloatValueOnShortName("r", 100);

		builder = new ConstantAnnealingScheduleBuilder(configuration);
		assertEquals(builder.getLearningRate(), 100, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 100, 0.);

		configuration.addFloatValueOnShortName("r", 155);
		builder.configure(configuration);

		assertEquals(builder.getLearningRate(), 155, 0.);
		schedule = builder.build();
		assertEquals(schedule.getInitialLearningRate(), 155, 0.);
	}
}
